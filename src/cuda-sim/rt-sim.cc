#include "ptx_ir.h"
#include "vector-math.h"
#include "rt-sim.h"

void decode_space( memory_space_t &space, ptx_thread_info *thread, const operand_info &op, memory_space *&mem, addr_t &addr);

void print_float4(float4 printVal) {
	printf("%f, %f, %f, %f\n", printVal.x, printVal.y, printVal.z, printVal.w);
}

void trace_ray(const class ptx_instruction * pI, class ptx_thread_info * thread, const class function_info * target_func, std::list<addr_t> & memory_accesses)
{
    unsigned n_return = target_func->has_return();
    assert(n_return == 0);
    unsigned n_args = target_func->num_args();
    assert(n_args == 3);
    // printf("Function has %d args.\n", n_args);
    
    int arg = 0;
    // First argument: Ray Properties
    const operand_info &actual_param_op1 = pI->operand_lookup(arg + 1);    
    const symbol *formal_param1 = target_func->get_arg(arg);
    addr_t from_addr = actual_param_op1.get_symbol()->get_address();
    unsigned size=formal_param1->get_size_in_bytes();
    assert(size == 32);
    
    // Ray
    Ray ray_properties;
    thread->m_local_mem->read(from_addr, size, &ray_properties);
    printf("Origin: (%f, %f, %f), Direction: (%f, %f, %f), tmin: %f, tmax: %f\n", 
      ray_properties.origin_tmin.x, ray_properties.origin_tmin.y, ray_properties.origin_tmin.z,
      ray_properties.dir_tmax.x, ray_properties.dir_tmax.y, ray_properties.dir_tmax.z,
      ray_properties.origin_tmin.w, ray_properties.dir_tmax.w);

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
    printf("Ray payload address: 0x%x\n", ray_payload_addr);
    
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
    printf("Node address: 0x%8x\n", node_start);
    
    // TODO: Figure out how to calculate triangle start (Modify rtao to use set offset?)
    // White Oak
    // addr_t tri_start = node_start + 0x11bc00; 
    // addr_t tri_end = tri_start + 0x221a10;
    // Teapot
    addr_t tri_start = node_start + 0x6e600; 
    addr_t tri_end = tri_start + 0xd6f40;
    
    // Global memory
    memory_space *mem=NULL;
    mem = thread->get_global_memory();
    
    
    // Traversal stack
    // const int STACK_SIZE = 32;
    // addr_t traversal_stack[STACK_SIZE];
    // traversal_stack[0] = EMPTY_STACK;
    // addr_t* stack_ptr = &traversal_stack[0];
    
    std::list<addr_t> traversal_stack;
    
    // Initialize
    addr_t child0_addr = 0;
    addr_t child1_addr = 0;
    addr_t next_node = 0;
    
    // Set thit to max
    float thit = ray_properties.dir_tmax.w;
    bool hit = false;
    addr_t hit_addr;
    
    do {

        // Check not a leaf node and not empty traversal stack (Leaf nodes start with 0xf...)
        while ((int)next_node >= 0)
        {
            if (next_node != 0) next_node *= 0x10;
            // Get top node data
            // const float4 n0xy = __ldg(localBVHTreeNodes + nodeAddr + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
            // const float4 n1xy = __ldg(localBVHTreeNodes + nodeAddr + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
            // const float4 n01z = __ldg(localBVHTreeNodes + nodeAddr + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
            float4 n0xy, n1xy, n01z;
            mem->read(node_start + next_node, sizeof(float4), &n0xy);
            mem->read(node_start + next_node + sizeof(float4), sizeof(float4), &n1xy);
            mem->read(node_start + next_node + 2*sizeof(float4), sizeof(float4), &n01z);
            
            // TODO: Figure out if node_start + next_node + 2 also should be recorded
            thread->add_raytrace_mem_access(node_start + next_node);
            memory_accesses.push_back(node_start + next_node);
            
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
            bool child0_hit = ray_box_test(n0lo, n0hi, ray_properties.get_direction(), ray_properties.get_origin(), ray_properties.get_tmin(), ray_properties.get_tmax(), thit0);
            bool child1_hit = ray_box_test(n1lo, n1hi, ray_properties.get_direction(), ray_properties.get_origin(), ray_properties.get_tmin(), ray_properties.get_tmax(), thit1);
            
            #ifdef DEBUG_PRINT
            printf("Child 0 hit: %d \t", child0_hit);
            printf("Child 1 hit: %d \n", child1_hit);
            #endif
            
            mem->read(node_start + next_node + 3*sizeof(float4), sizeof(addr_t), &child0_addr);
            mem->read(node_start + next_node + 3*sizeof(float4) + sizeof(addr_t), sizeof(addr_t), &child1_addr);
            
            #ifdef DEBUG_PRINT
            printf("Child 0 offset: 0x%x \t", child0_addr);
            printf("Child 1 offset: 0x%x \n", child1_addr);
            #endif
            
            
            // Miss
            if (!child0_hit && !child1_hit) {
                if (traversal_stack.empty()) {
                    next_node = EMPTY_STACK;
                    break;
                }
                
                // Pop next node from stack
                next_node = traversal_stack.back();
                traversal_stack.pop_back();
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
            memory_accesses.push_back(tri_start + tri_addr);

            // Triangle intersection algorithm
            hit = mt_ray_triangle_test(p0, p1, p2, ray_properties, &thit);
            
            #else 
            
            // while triangle address is within triangle primitive range
            while (tri_addr <= (tri_end - tri_start)) {
                
                // Matches rtao Woopify triangle
                float4 p0, p1, p2;
                mem->read(tri_start + tri_addr, sizeof(float4), &p0);
                mem->read(tri_start + tri_addr + sizeof(float4), sizeof(float4), &p1);
                mem->read(tri_start + tri_addr + 2*sizeof(float4), sizeof(float4), &p2);
                thread->add_raytrace_mem_access(tri_start + tri_addr);
                memory_accesses.push_back(tri_start + tri_addr);
                
                // Check if triangle is valid (if (__float_as_int(v00.x) == 0x80000000))
                if (*(int*)&p0.x ==  0x80000000) {
                    #ifdef DEBUG_PRINT
                    printf("End of primitives in leaf node. \n");
                    #endif
                    break;
                }
                
                hit = rtao_ray_triangle_test(p0, p1, p2, ray_properties, &thit, ray_payload);
                if (hit) {
                    #ifdef DEBUG_PRINT
                    printf("HIT\t t: %f\n", thit);
                    #endif
                    ray_payload.t_triId_u_v.y = tri_addr >> 4;
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
            #ifdef DEBUG_PRINT
            printf("Traversal Stack: \n");
            print_stack(traversal_stack);
            #endif
        }
        
    }  while (next_node != EMPTY_STACK);
    
    if (thit != ray_properties.get_tmax()) {
        printf("\nResult: (t, addr, u, v)\n");
        printf("t: %f, u: %f, v: %f, triangle offset: 0x%x\n", ray_payload.t_triId_u_v.x, ray_payload.t_triId_u_v.z, ray_payload.t_triId_u_v.w, (addr_t)ray_payload.t_triId_u_v.y);
        
        mem->write(ray_payload_addr, sizeof(Hit), &ray_payload, NULL, NULL);
        
        // TODO: Keep this separate from read accesses
        thread->add_raytrace_mem_access(ray_payload_addr);
    }
    
    printf("Memory Accesses:\n");
    print_stack(memory_accesses);
}


bool ray_box_test(float3 low, float3 high, float3 direction, float3 origin, float tmin, float tmax, float& thit)
{
	// const float3 lo = Low * InvDir - Ood;
	// const float3 hi = High * InvDir - Ood;
    float3 lo = get_t_bound(low, origin, direction);
    float3 hi = get_t_bound(high, origin, direction);
    
    // QUESTION: max value does not match rtao benchmark, rtao benchmark converts float to int with __float_as_int
    // i.e. __float_as_int: -110.704826 => -1025677090, -24.690834 => -1044019502
    
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

float3 get_t_bound(float3 box, float3 origin, float3 direction)
{
    // Avoid div by zero, returns 1/2^80, an extremely small number
    const float ooeps = exp2f(-80.0f); 
    
    // Calculate inverse direction
    float3 idir;
    idir.x = 1.0f / (fabsf(direction.x) > ooeps ? direction.x : copysignf(ooeps, direction.x));
    idir.y = 1.0f / (fabsf(direction.y) > ooeps ? direction.y : copysignf(ooeps, direction.y));
    idir.z = 1.0f / (fabsf(direction.z) > ooeps ? direction.z : copysignf(ooeps, direction.z));
    
    // Calculate bounds
    float3 result;
    result.x = (box.x - origin.x) * idir.x;
    result.y = (box.y - origin.y) * idir.y;
    result.z = (box.z - origin.z) * idir.z;
    
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