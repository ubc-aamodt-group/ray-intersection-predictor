#include "ptx_ir.h"
#include "vector-math.h"
#include "rt-sim.h"
#include "../../libcuda/gpgpu_context.h"

typedef uint64_t(*HashFunc)(const Ray&, const float3&, const float3&);

void decode_space( memory_space_t &space, ptx_thread_info *thread, const operand_info &op, memory_space *&mem, addr_t &addr);

void print_float4(float4 printVal) {
	printf("%f, %f, %f, %f\n", printVal.x, printVal.y, printVal.z, printVal.w);
}

void trace_ray(const class ptx_instruction * pI, class ptx_thread_info * thread, const class function_info * target_func)
{    
    unsigned n_return = target_func->has_return();
    assert(n_return == 0);
    unsigned n_args = target_func->num_args();
    assert(n_args == 4);
    // printf("Function has %d args.\n", n_args);
    
    int arg = 0;
    // First argument: Ray Properties
    const operand_info &actual_param_op1 = pI->operand_lookup(arg + 1);    
    const symbol *formal_param1 = target_func->get_arg(arg);
    addr_t from_addr = actual_param_op1.get_symbol()->get_address();
    unsigned size=formal_param1->get_size_in_bytes();
    assert(size == sizeof(Ray));
    
    // Ray
    Ray ray_properties;
    thread->m_local_mem->read(from_addr, size, &ray_properties);
    #ifdef DEBUG_PRINT
    printf("Origin: (%f, %f, %f), Direction: (%f, %f, %f), tmin: %f, tmax: %f\n", 
      ray_properties.origin_tmin.x, ray_properties.origin_tmin.y, ray_properties.origin_tmin.z,
      ray_properties.dir_tmax.x, ray_properties.dir_tmax.y, ray_properties.dir_tmax.z,
      ray_properties.origin_tmin.w, ray_properties.dir_tmax.w);
    #endif

    arg++;
    // Second argument: Ray Payload
    const operand_info &actual_param_op2 = pI->operand_lookup(arg + 1);    
    const symbol *formal_param2 = target_func->get_arg(arg);
    from_addr = actual_param_op2.get_symbol()->get_address();
    size=formal_param2->get_size_in_bytes();  
    assert(size == 8);
    
    // Payload
    Hit ray_payload;
    addr_t ray_payload_addr;
    thread->m_local_mem->read(from_addr, size, &ray_payload_addr);
    #ifdef DEBUG_PRINT
    printf("Ray payload address: 0x%x\n", ray_payload_addr);
    #endif
    
    arg++;
    // Third argument: Top of BVH Tree
    const operand_info &actual_param_op = pI->operand_lookup(arg + 1);    
    const symbol *formal_param = target_func->get_arg(arg);
    from_addr = actual_param_op.get_symbol()->get_address();
    size=formal_param->get_size_in_bytes();
    assert(size == 8);
    
    // Top node
    addr_t node_start;
    thread->m_local_mem->read(from_addr, size, &node_start);
    #ifdef DEBUG_PRINT
    printf("Node address: 0x%8x\n", node_start);
    #endif
    
    arg++;
    // Fourth argument: Start of triangle data
    const operand_info &actual_param_op4 = pI->operand_lookup(arg + 1);    
    const symbol *formal_param4 = target_func->get_arg(arg);
    from_addr = actual_param_op4.get_symbol()->get_address();
    size=formal_param4->get_size_in_bytes();
    
    assert(size == 8);
    addr_t tri_start;
    thread->m_local_mem->read(from_addr, size, &tri_start);

    thread->set_node_start(node_start);
    thread->set_tri_start(tri_start);
    
    // Global memory
    memory_space *mem=NULL;
    mem = thread->get_global_memory();
    
    // Compute world bounds, which is just the bounding box of the first BVH node
    float3 world_min, world_max;
    {
      float4 n0xy, n1xy, n01z;
      mem->read(node_start, sizeof(float4), &n0xy);
      mem->read(node_start + sizeof(float4), sizeof(float4), &n1xy);
      mem->read(node_start + 2*sizeof(float4), sizeof(float4), &n01z);

      float3 n0lo, n0hi, n1lo, n1hi;
      n0lo = {n0xy.x, n0xy.z, n01z.x};
      n0hi = {n0xy.y, n0xy.w, n01z.y};
      n1lo = {n1xy.x, n1xy.z, n01z.z};
      n1hi = {n1xy.y, n1xy.w, n01z.w};
      world_min = min(n0lo, n1lo);
      world_max = max(n0hi, n1hi);
    }

    // Compute ray hash
    std::vector<unsigned long long> ray_hashes;
    ray_hashes = get_ray_hashes(ray_properties, world_min, world_max);
    thread->add_ray_hashes(ray_hashes);
    thread->add_ray_properties(ray_properties);
    
    std::list<addr_t> traversal_stack;
    std::vector<addr_t> nodes_stack;
    
    // Initialize
    addr_t child0_addr = 0;
    addr_t child1_addr = 0;
    addr_t next_node = 0;
    addr_t predict_node = 0;
    
    // Set thit to max
    float thit = ray_properties.dir_tmax.w;
    bool hit = false;
    addr_t hit_addr;
    
    // Get predictor config
    ray_predictor_config predictor_config = GPGPU_Context()->the_gpgpusim->g_the_gpu->get_config().get_ray_predictor_config();
    
    // Map of address to tree level
    std::map<new_addr_type, unsigned> tree_level_map;
    tree_level_map[node_start] = 1;

    std::map<unsigned long long, int> tree_depth_map;
    tree_depth_map[0] = 0;
    
    unsigned max_tree_depth = 0;

    int num_nodes_accessed = 0;
    int num_triangles_accessed = 0;
        
    do {

        // Check not a leaf node and not empty traversal stack (Leaf nodes start with 0xf...)
        while ((int)next_node >= 0)
        {
            nodes_stack.push_back(next_node);

            if (next_node != 0) next_node *= 0x10;
            // Get top node data
            // const float4 n0xy = __ldg(localBVHTreeNodes + nodeAddr + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
            // const float4 n1xy = __ldg(localBVHTreeNodes + nodeAddr + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
            // const float4 n01z = __ldg(localBVHTreeNodes + nodeAddr + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
            float4 n0xy, n1xy, n01z;
            mem->read(node_start + next_node, sizeof(float4), &n0xy);
            mem->read(node_start + next_node + sizeof(float4), sizeof(float4), &n1xy);
            mem->read(node_start + next_node + 2*sizeof(float4), sizeof(float4), &n01z);
            
            unsigned current_tree_level = tree_level_map[node_start + next_node];
            assert(current_tree_level > 0);
            
            if (current_tree_level > max_tree_depth) {
                max_tree_depth = current_tree_level;
            }
            
            thread->add_raytrace_mem_access(node_start + next_node);
            GPGPU_Context()->func_sim->g_total_raytrace_node_accesses++;
            num_nodes_accessed++;
            
            if (GPGPU_Context()->func_sim->g_raytrace_addresses.find(node_start + next_node) == GPGPU_Context()->func_sim->g_raytrace_addresses.end()) {
                GPGPU_Context()->func_sim->g_unique_node_accesses++;
            }
            GPGPU_Context()->func_sim->g_raytrace_addresses.insert(node_start + next_node);
            
            #ifdef DEBUG_PRINT
            printf("Node data: \n");
            print_float4(n0xy);
            print_float4(n1xy);
            print_float4(n01z);
            #endif
            
            // Reorganize
            float3 n0lo, n0hi, n1lo, n1hi;
            n0lo = {n0xy.x, n0xy.z, n01z.x};
            n0hi = {n0xy.y, n0xy.w, n01z.y};
            n1lo = {n1xy.x, n1xy.z, n01z.z};
            n1hi = {n1xy.y, n1xy.w, n01z.w};
            
            float thit0, thit1;
            float3 idir = calculate_idir(ray_properties.get_direction());
            bool child0_hit = ray_box_test(n0lo, n0hi, idir, ray_properties.get_origin(), ray_properties.get_tmin(), ray_properties.get_tmax(), thit0);
            bool child1_hit = ray_box_test(n1lo, n1hi, idir, ray_properties.get_origin(), ray_properties.get_tmin(), ray_properties.get_tmax(), thit1);
            
            #ifdef DEBUG_PRINT
            printf("Child 0 hit: %d \t", child0_hit);
            printf("Child 1 hit: %d \n", child1_hit);
            #endif
            
            mem->read(node_start + next_node + 3*sizeof(float4), sizeof(addr_t), &child0_addr);
            mem->read(node_start + next_node + 3*sizeof(float4) + sizeof(addr_t), sizeof(addr_t), &child1_addr);

            // Assume we store the go-up ancestor in the padded space
            // Note: this value isn't actually used. We keep track of the go-up ancestor using the
            // nodes_stack so that the BVH doesn't need to change for different go-up levels.
            addr_t go_up_addr;
            mem->read(node_start + next_node + 3*sizeof(float4) + 2 * sizeof(addr_t), sizeof(addr_t), &go_up_addr);
            
            if ((int)child0_addr > 0)
                tree_level_map[node_start + child0_addr * 0x10] = current_tree_level + 1;
            if ((int)child1_addr > 0)
                tree_level_map[node_start + child1_addr * 0x10] = current_tree_level + 1;

            tree_depth_map[child0_addr * 0x10] = tree_depth_map[next_node] + 1;
            tree_depth_map[child1_addr * 0x10] = tree_depth_map[next_node] + 1;
            
            #ifdef DEBUG_PRINT
            printf("Child 0 offset: 0x%x \t", child0_addr);
            printf("Child 1 offset: 0x%x \n", child1_addr);
            #endif
            
            
            // Miss
            if (!child0_hit && !child1_hit) {
                nodes_stack.pop_back();

                if (traversal_stack.empty()) {
                    next_node = EMPTY_STACK;
                    break;
                }
                
                // Pop next node from stack
                next_node = traversal_stack.back();
                traversal_stack.pop_back();
                nodes_stack.resize(tree_depth_map[next_node * 0x10]);
                #ifdef DEBUG_PRINT
                printf("Traversal Stack: \n");
                print_stack(traversal_stack);
                #endif
            }
            // Both hit
            else if (child0_hit && child1_hit) {
                next_node = (thit0 < thit1) ? child0_addr : child1_addr;
                
                // Push extra node to stack
                traversal_stack.push_back((thit0 < thit1) ? child1_addr : child0_addr);
                #ifdef DEBUG_PRINT
                printf("Traversal Stack: \n");
                print_stack(traversal_stack);
                #endif
                
                if (traversal_stack.size() > MAX_TRAVERSAL_STACK_SIZE) printf("Short stack full!\n");
            }
            // Single hit
            else {
                assert(child0_hit ^ child1_hit);
                next_node = (child0_hit) ? child0_addr : child1_addr;
            }
            
        }
        #ifdef DEBUG_PRINT
        printf("Transition to leaf nodes.\n");
        #endif
        
        addr_t tri_addr;

        
        while ((int)next_node < 0) {
            
            // Convert to triangle offset
            tri_addr = ~next_node;
            tri_addr *= 0x10;
            
            
            // Load vertices
            #ifdef MOLLER_TRUMBORE
                
            float3 p0, p1, p2;
            mem->read(tri_start + tri_addr, sizeof(float3), &p0);
            mem->read(tri_start + tri_addr + sizeof(float3), sizeof(float3), &p1);
            mem->read(tri_start + tri_addr + 2*sizeof(float3), sizeof(float3), &p2);
            thread->add_raytrace_mem_access(tri_start + tri_addr);

            // Triangle intersection algorithm
            hit = mt_ray_triangle_test(p0, p1, p2, ray_properties, &thit);
            
            #else 
            
            // while triangle address is within triangle primitive range
            while (1) {
                
                // Matches rtao Woopify triangle
                float4 p0, p1, p2;
                mem->read(tri_start + tri_addr, sizeof(float4), &p0);
                mem->read(tri_start + tri_addr + sizeof(float4), sizeof(float4), &p1);
                mem->read(tri_start + tri_addr + 2*sizeof(float4), sizeof(float4), &p2);
                
                // Check if triangle is valid (if (__float_as_int(v00.x) == 0x80000000))
                if (*(int*)&p0.x ==  0x80000000) {
                    #ifdef DEBUG_PRINT
                    printf("End of primitives in leaf node. \n");
                    #endif
                    break;
                }
                
                thread->add_raytrace_mem_access(tri_start + tri_addr);
                GPGPU_Context()->func_sim->g_total_raytrace_triangle_accesses++;
                num_triangles_accessed++;
                
                if (GPGPU_Context()->func_sim->g_raytrace_addresses.find(tri_start + tri_addr) == GPGPU_Context()->func_sim->g_raytrace_addresses.end()) {
                    GPGPU_Context()->func_sim->g_unique_triangle_accesses++;
                }
                GPGPU_Context()->func_sim->g_raytrace_addresses.insert(tri_start + tri_addr);
                
                tree_level_map[tri_start + tri_addr] = 0xff;
                
                hit = rtao_ray_triangle_test(p0, p1, p2, ray_properties, &thit, ray_payload);
                if (hit) {
                    #ifdef DEBUG_PRINT
                    printf("HIT\t t: %f\n", thit);
                    #endif
                    *(int*) &ray_payload.t_triId_u_v.y = tri_addr >> 4;
                    
                    predict_node = next_node;
                    if (predictor_config.go_up_level > 0) {
                        size_t go_up_index =
                            std::max(0, (int) nodes_stack.size() - (int) predictor_config.go_up_level);
                        predict_node = nodes_stack[go_up_index];

                        assert(
                            tree_depth_map[next_node * 0x10] - tree_depth_map[predict_node * 0x10] == nodes_stack.size() - go_up_index);
                    }

                    if (ray_properties.anyhit) {
                        traversal_stack.clear();
                        next_node = EMPTY_STACK;
                        break;
                    }
                }
                
                #ifdef DEBUG_PRINT
                else
                    printf("MISS\n");
                #endif
                    
                // Go to next triangle
                tri_addr += 0x30;
            
            }
            
            #endif

            if (traversal_stack.empty()) {
                next_node = EMPTY_STACK;
                break;
            }
            // Pop next node off stack
            next_node = traversal_stack.back();
            traversal_stack.pop_back();
            nodes_stack.resize(tree_depth_map[next_node * 0x10]);
            #ifdef DEBUG_PRINT
            printf("Traversal Stack: \n");
            print_stack(traversal_stack);
            #endif
        }
        
    }  while (next_node != EMPTY_STACK);
    
    if (thit != ray_properties.get_tmax()) {
        #ifdef DEBUG_PRINT
        printf("\nResult: (t, addr, u, v)\n");
        printf("t: %f, u: %f, v: %f, triangle offset: 0x%x\n", ray_payload.t_triId_u_v.x, ray_payload.t_triId_u_v.z, ray_payload.t_triId_u_v.w, (addr_t)ray_payload.t_triId_u_v.y);
        #endif
        
        mem->write(ray_payload_addr, sizeof(Hit), &ray_payload, NULL, NULL);
        
        thread->add_ray_intersect();
        thread->add_ray_prediction(predict_node);
    } else{
        *(int*) &ray_payload.t_triId_u_v.y = (int)-1;
        mem->write(ray_payload_addr, sizeof(Hit), &ray_payload, NULL, NULL);
    }

    thread->set_tree_level_map(tree_level_map);
    thread->set_num_nodes_accessed(num_nodes_accessed);
    thread->set_num_triangles_accessed(num_triangles_accessed);
    
    if (GPGPU_Context()->func_sim->g_max_tree_depth.find(max_tree_depth) == GPGPU_Context()->func_sim->g_max_tree_depth.end()) {
        GPGPU_Context()->func_sim->g_max_tree_depth[max_tree_depth] = 0;
    }
    GPGPU_Context()->func_sim->g_max_tree_depth[max_tree_depth]++;
}

bool ray_box_test(float3 low, float3 high, float3 idirection, float3 origin, float tmin, float tmax, float& thit)
{
	// const float3 lo = Low * InvDir - Ood;
	// const float3 hi = High * InvDir - Ood;
    float3 lo = get_t_bound(low, origin, idirection);
    float3 hi = get_t_bound(high, origin, idirection);
    
	// const float slabMin = tMinFermi(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, TMin);
	// const float slabMax = tMaxFermi(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, TMax);
    float min = magic_max7(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, tmin);
    float max = magic_min7(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, tmax);

	// OutIntersectionDist = slabMin;
    thit = min;

	// return slabMin <= slabMax;
    return (min <= max);
}

bool mt_ray_triangle_test(float3 p0, float3 p1, float3 p2, Ray ray_properties, float* thit)
{
    // Moller Trumbore algorithm (from scratchapixel.com)
    float3 v0v1 = p1 - p0;
    float3 v0v2 = p2 - p0;
    float3 pvec = cross(ray_properties.get_direction(), v0v2);
    float det = dot(v0v1, pvec);
    
    float idet = 1 / det;
    
    float3 tvec = ray_properties.get_origin() - p0;
    float u = dot(tvec, pvec) * idet;
    
    if (u < 0 || u > 1) return false;
    
    float3 qvec = cross(tvec, v0v1);
    float v = dot(ray_properties.get_direction(), qvec) * idet;
    
    if (v < 0 || (u + v) > 1) return false;
    
    *thit = dot(v0v2, qvec) * idet;
    return true;
}

bool rtao_ray_triangle_test(float4 v00, float4 v11, float4 v22, Ray ray_properties, float* thit) {
    Hit ray_payload;
    return rtao_ray_triangle_test(v00, v11, v22, ray_properties, thit, ray_payload);    
}

bool rtao_ray_triangle_test(float4 v00, float4 v11, float4 v22, Ray ray_properties, float* thit, Hit &ray_payload)
{
    
	float Oz = v00.w - ray_properties.get_origin().x * v00.x - ray_properties.get_origin().y * v00.y - ray_properties.get_origin().z * v00.z;
	float invDz = 1.0f / (ray_properties.get_direction().x*v00.x + ray_properties.get_direction().y*v00.y + ray_properties.get_direction().z*v00.z);
	float t = Oz * invDz;

	if (t > ray_properties.get_tmin() && t < *thit) {
		float Ox = v11.w + ray_properties.get_origin().x * v11.x + ray_properties.get_origin().y * v11.y + ray_properties.get_origin().z * v11.z;
		float Dx = ray_properties.get_direction().x * v11.x + ray_properties.get_direction().y * v11.y + ray_properties.get_direction().z * v11.z;
		float u = Ox + t * Dx;

		if (u >= 0.0f && u <= 1.0f) {
			float Oy = v22.w + ray_properties.get_origin().x * v22.x + ray_properties.get_origin().y * v22.y + ray_properties.get_origin().z * v22.z;
			float Dy = ray_properties.get_direction().x * v22.x + ray_properties.get_direction().y * v22.y + ray_properties.get_direction().z * v22.z;
			float v = Oy + t*Dy;

			if (v >= 0.0f && u + v <= 1.0f) {
				// triangleuv.x = u;
				// triangleuv.y = v;

				*thit = t;
                ray_payload.t_triId_u_v = {t, 0x0, u, v};
                return true;
			}
		}
	}

    return false;
}

float3 get_t_bound(float3 box, float3 origin, float3 idirection)
{
    // // Avoid div by zero, returns 1/2^80, an extremely small number
    // const float ooeps = exp2f(-80.0f); 
    
    // // Calculate inverse direction
    // float3 idir;
    // idir.x = 1.0f / (fabsf(direction.x) > ooeps ? direction.x : copysignf(ooeps, direction.x));
    // idir.y = 1.0f / (fabsf(direction.y) > ooeps ? direction.y : copysignf(ooeps, direction.y));
    // idir.z = 1.0f / (fabsf(direction.z) > ooeps ? direction.z : copysignf(ooeps, direction.z));
    
    // Calculate bounds
    float3 result;
    result.x = (box.x - origin.x) * idirection.x;
    result.y = (box.y - origin.y) * idirection.y;
    result.z = (box.z - origin.z) * idirection.z;
    
    // Return
    return result;
}

float magic_max7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
	float t1 = MIN_MAX(a0, a1, d);
	float t2 = MIN_MAX(b0, b1, t1);
	float t3 = MIN_MAX(c0, c1, t2);
	return t3;
}

float magic_min7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
	float t1 = MAX_MIN(a0, a1, d);
	float t2 = MAX_MIN(b0, b1, t1);
	float t3 = MAX_MIN(c0, c1, t2);
	return t3;
}

void print_stack(std::list<addr_t> &traversal_stack)
{
    if (traversal_stack.empty()) printf("Empty!\n");
    else{
        for(std::list<addr_t>::iterator iter = traversal_stack.begin(); iter != traversal_stack.end(); iter++){
            printf("0x%x\t", *iter);
        }
        printf("\n");
    }
}

unsigned bfind(unsigned a) {
    unsigned int d = 0xffffffff;
    for (int i = 31; i>=0; i--) {
        if (a & (1<<i)) {
            d = i;
            break;
        }
    }
    return d;
}

unsigned popc(unsigned a) {
    unsigned d = 0;
    while (a != 0) {
        if ( a & 0x1) d++;
        a = a >> 1;
    } 
    return d;
}

float uint_as_float(unsigned int a) {
    return *(float *)&a;
}

float3 calculate_idir(float3 direction) {
    // Avoid div by zero, returns 1/2^80, an extremely small number
    const float ooeps = exp2f(-80.0f); 
    
    // Calculate inverse direction
    float3 idir;
    idir.x = 1.0f / (fabsf(direction.x) > ooeps ? direction.x : copysignf(ooeps, direction.x));
    idir.y = 1.0f / (fabsf(direction.y) > ooeps ? direction.y : copysignf(ooeps, direction.y));
    idir.z = 1.0f / (fabsf(direction.z) > ooeps ? direction.z : copysignf(ooeps, direction.z));
    
    return idir;    
}



uint64_t hash_comp(float x, uint32_t num_bits) {
    uint32_t mask = UINT32_MAX >> (32 - num_bits);

    uint32_t o_x = *((uint32_t*) &x);

    uint64_t sign_bit_x = o_x >> 31;
    uint64_t exp_x = (o_x >> (31 - num_bits)) & mask;
    uint64_t mant_x = (o_x >> (23 - num_bits)) & mask;

    return (sign_bit_x << (2 * num_bits)) | (exp_x << num_bits) | mant_x;
}

uint64_t hash_francois(const Ray &ray, uint32_t num_bits) {
  // Each component has 1 bit sign, `num_bits` mantissa, `num_bits` exponent
  uint32_t num_comp_bits = 2 * num_bits + 1;
  uint64_t hash_d =
    (hash_comp(ray.get_direction().z, num_bits) << (2 * num_comp_bits)) |
    (hash_comp(ray.get_direction().y, num_bits) << num_comp_bits) |
     hash_comp(ray.get_direction().x, num_bits);
  uint64_t hash_o =
    (hash_comp(ray.get_origin().x, num_bits) << (2 * num_comp_bits)) |
    (hash_comp(ray.get_origin().y, num_bits) << num_comp_bits) |
     hash_comp(ray.get_origin().z, num_bits);
  return hash_o ^ hash_d;
}

// Quantize direction to a sphere - xyz to theta and phi
// `theta_bits` is used for theta, `theta_bits` + 1 is used for phi, for a total of
// 2 * `theta_bits` + 1 bits
uint64_t hash_direction_spherical(const float3 &d, uint32_t num_sphere_bits) {
  uint32_t theta_bits = num_sphere_bits;
  uint32_t phi_bits = theta_bits + 1;

  uint64_t theta = std::acos(clamp(d.z, -1.f, 1.f)) / PI * 180;
  uint64_t phi = (std::atan2(d.y, d.x) + PI) / PI * 180;
  uint64_t q_theta = theta >> (8 - theta_bits);
  uint64_t q_phi = phi >> (9 - phi_bits);

  return (q_phi << theta_bits) | q_theta;
}

// Quantize origin to a grid
// Each component uses `num_bits`, for a total of 3 * `num_bits` bits
uint64_t hash_origin_grid(const float3& o, const float3& min,
                          const float3& max, uint32_t num_bits) {
  uint32_t grid_size = 1 << num_bits;

  uint64_t hash_o_x = clamp((o.x - min.x) / (max.x - min.x) * grid_size, 0.f, (float)grid_size - 1);
  uint64_t hash_o_y = clamp((o.y - min.y) / (max.y - min.y) * grid_size, 0.f, (float)grid_size - 1);
  uint64_t hash_o_z = clamp((o.z - min.z) / (max.z - min.z) * grid_size, 0.f, (float)grid_size - 1);
  return (hash_o_x << (2 * num_bits)) | (hash_o_y << num_bits) | hash_o_z;
}

uint64_t hash_grid_spherical(const Ray &ray, const float3& min, const float3& max,
                             uint32_t num_grid_bits, uint32_t num_sphere_bits)
{
  uint64_t hash_d = hash_direction_spherical(ray.get_direction(), num_sphere_bits);
  uint64_t hash_o = hash_origin_grid(ray.get_origin(), min, max, num_grid_bits);
  uint64_t hash = hash_o ^ hash_d;

  return hash;
}

uint64_t hash_francois_grid_spherical(const Ray &ray,
                                      const float3& min,
                                      const float3& max,
                                      uint32_t num_francois_bits,
                                      uint32_t num_grid_bits,
                                      uint32_t num_sphere_bits) {
  return hash_grid_spherical(ray, min, max, num_grid_bits, num_sphere_bits) ^
         hash_francois(ray, num_francois_bits);
}

uint64_t hash_two_point(const Ray &ray,
                        const float3& min,
                        const float3& max,
                        uint32_t num_grid_bits,
                        float est_length_ratio) {
  uint64_t hash_1 = hash_origin_grid(ray.get_origin(), min, max, num_grid_bits);
  float3 d = max - min;
  float max_extent_length = std::max(std::max(d.x, d.y), d.z);
  float3 est_target = ray.get_origin() + est_length_ratio * max_extent_length * ray.get_direction();
  uint64_t hash_2 = hash_origin_grid(est_target, min, max, num_grid_bits);
  return hash_1 ^ hash_2;
}

const std::vector<HashFunc>& get_hash_functions() {
  static std::vector<HashFunc> hash_funcs;

  if (!hash_funcs.empty()) {
    return hash_funcs;
  }

  const ray_predictor_config& config =
    GPGPU_Context()->the_gpgpusim->g_the_gpu->get_config().get_ray_predictor_config();

  // Hash type
  bool hash_use_francois = config.hash_use_francois;
  bool hash_use_grid_spherical = config.hash_use_grid_spherical;
  bool hash_use_two_point = config.hash_use_two_point;

  if (hash_use_francois) {
    hash_funcs.push_back(
      [](const Ray& ray, const float3& min, const float3& max) -> uint64_t {
        const ray_predictor_config& config =
          GPGPU_Context()->the_gpgpusim->g_the_gpu->get_config().get_ray_predictor_config();
        return hash_francois(ray, config.hash_francois_bits);
      }
    );
  }
  if (hash_use_grid_spherical) {
    hash_funcs.push_back(
      [](const Ray& ray, const float3& min, const float3& max) {
        const ray_predictor_config& config =
          GPGPU_Context()->the_gpgpusim->g_the_gpu->get_config().get_ray_predictor_config();
        return hash_grid_spherical(ray, min, max, config.hash_grid_bits, config.hash_sphere_bits);
      }
    );
  }
  if (hash_use_two_point) {
    hash_funcs.push_back(
      [](const Ray& ray, const float3& min, const float3& max) -> uint64_t {
        const ray_predictor_config& config =
          GPGPU_Context()->the_gpgpusim->g_the_gpu->get_config().get_ray_predictor_config();
        return hash_two_point(ray, min, max, config.hash_grid_bits, config.hash_two_point_est_length_ratio);
      }
    );
  }

  return hash_funcs;
}

std::vector<unsigned long long> get_ray_hashes(const Ray &ray, const float3& min, const float3& max) {
  std::vector<unsigned long long> ray_hashes;

  const std::vector<HashFunc>& hash_funcs = get_hash_functions();
  for (HashFunc func : hash_funcs) {
    uint64_t hash = func(ray, min, max);
    if (std::find(ray_hashes.begin(), ray_hashes.end(), hash) == ray_hashes.end()) {
      ray_hashes.push_back(hash);
    }
  }

  return ray_hashes;
}