#include "ptx_ir.h"
#include "rt-sim.h"

void decode_space( memory_space_t &space, ptx_thread_info *thread, const operand_info &op, memory_space *&mem, addr_t &addr);

void print_float4(float4 printVal) {
	printf("%f, %f, %f, %f\n", printVal.x, printVal.y, printVal.z, printVal.w);
}

void trace_ray(const class ptx_instruction * pI, class ptx_thread_info * thread, const class function_info * target_func )
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
    assert(size == 16);
    
    // Payload
    Hit ray_payload;
    thread->m_local_mem->read(from_addr, size, &ray_payload);
    print_float4(ray_payload.t_triId_u_v);
    
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
    
    // TODO: Figure out how triangle start address can be calculated
    addr_t tri_start = node_start + 0x6e5c0; 
    
    // Global memory
    memory_space *mem=NULL;
    mem = thread->get_global_memory();
    
    
    // Traversal stack
    const int STACK_SIZE = 32;
    addr_t traversal_stack[STACK_SIZE];
    traversal_stack[0] = EMPTY_STACK;
    addr_t* stack_ptr = &traversal_stack[0];
    
    // Initialize
    addr_t child0_addr = 0;
    addr_t child1_addr = 0;
    addr_t next_node = 0;
    
    // Set thit to max
    float thit = ray_properties.dir_tmax.w;
    bool hit = false;
       
    while (next_node != EMPTY_STACK) {

        // Check not a leaf node and not empty traversal stack (Leaf nodes start with 0xf...)
        while ((int)next_node >= 0 && next_node != EMPTY_STACK)
        {
            // Get top node data
            // const float4 n0xy = __ldg(localBVHTreeNodes + nodeAddr + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
            // const float4 n1xy = __ldg(localBVHTreeNodes + nodeAddr + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
            // const float4 n01z = __ldg(localBVHTreeNodes + nodeAddr + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
            float4 n0xy, n1xy, n01z;
            mem->read(node_start + next_node, sizeof(float4), &n0xy);
            mem->read(node_start + next_node + sizeof(float4), sizeof(float4), &n1xy);
            mem->read(node_start + next_node + 2*sizeof(float4), sizeof(float4), &n01z);
            
            printf("Node data: \n");
            print_float4(n0xy);
            print_float4(n1xy);
            print_float4(n01z);
            
            // Reorganize
            float3 n0lo, n0hi, n1lo, n1hi;
            n0lo = {n0xy.x, n0xy.z, n01z.x};
            n0hi = {n0xy.y, n0xy.w, n01z.y};
            n1lo = {n1xy.x, n1xy.z, n01z.z};
            n1hi = {n1xy.y, n1xy.w, n01z.w};
            
            float thit0, thit1;
            bool child0_hit = ray_box_test(n0lo, n0hi, ray_properties.get_direction(), ray_properties.get_origin(), ray_properties.get_tmin(), ray_properties.get_tmax(), thit0);
            bool child1_hit = ray_box_test(n1lo, n1hi, ray_properties.get_direction(), ray_properties.get_origin(), ray_properties.get_tmin(), ray_properties.get_tmax(), thit1);
            
            printf("Child 0 hit: %d \t", child0_hit);
            printf("Child 1 hit: %d \n", child1_hit);
            
            mem->read(node_start + next_node + 3*sizeof(float4), sizeof(addr_t), &child0_addr);
            child0_addr *= 0x10;
            mem->read(node_start + next_node + 3*sizeof(float4) + sizeof(addr_t), sizeof(addr_t), &child1_addr);
            child1_addr *= 0x10;
            
            printf("Child 0 offset: 0x%x \t", child0_addr);
            printf("Child 1 offset: 0x%x \n", child1_addr);
            
            
            // Miss
            if (!child0_hit && !child1_hit) {
                
                // Pop next node from stack
                next_node = *stack_ptr;
                stack_ptr--;
                print_stack(stack_ptr, traversal_stack);
                
            }
            // Both hit
            else if (child0_hit && child1_hit) {
                next_node = (thit0 < thit1) ? child0_addr : child1_addr;
                
                // Push extra node to stack
                stack_ptr++;
                *stack_ptr = (thit0 < thit1) ? child1_addr : child0_addr;
                print_stack(stack_ptr, traversal_stack);
            }
            // Single hit
            else {
                assert(child0_hit ^ child1_hit);
                next_node = (child0_hit) ? child0_addr : child1_addr;
            }
            
        }
        printf("Transition to leaf nodes.\n");
        
        addr_t tri_addr;

        
        while ((int)next_node < 0) {
            
            // Convert to triangle offset
            tri_addr = ~next_node;
            
            // Load vertices
            // TODO: Convert vertices in BVH to float3?
            float3 p0, p1, p2;
            mem->read(tri_start + tri_addr, sizeof(float3), &p0);
            mem->read(tri_start + tri_addr + sizeof(float3), sizeof(float3), &p1);
            mem->read(tri_start + tri_addr + 2*sizeof(float3), sizeof(float3), &p2);

            // Triangle intersection algorithm
            hit = ray_triangle_test(p0, p1, p2, ray_properties, &thit);

            // Pop next node off stack
            next_node = *stack_ptr;
            stack_ptr--;
        }
    }
    
    // TODO: Store hit into ray payload
        
}


bool ray_box_test(float3 low, float3 high, float3 direction, float3 origin, float tmin, float tmax, float& thit)
{
	// const float3 lo = Low * InvDir - Ood;
	// const float3 hi = High * InvDir - Ood;
    float3 lo = get_t_bound(low, origin, direction);
    float3 hi = get_t_bound(high, origin, direction);
    
    // TODO: max value does not match rtao benchmark
	// const float slabMin = tMinFermi(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, TMin);
	// const float slabMax = tMaxFermi(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, TMax);
    float min = magic_max7(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, tmin);
    float max = magic_min7(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, tmax);

	// OutIntersectionDist = slabMin;
    thit = min;

	// return slabMin <= slabMax;
    return (min <= max);
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

void print_stack(addr_t* stack_ptr, addr_t* traversal_stack) 
{
    printf("Traversal Stack: \n");
    for (addr_t* i=traversal_stack; i<=stack_ptr; i++) {
        printf("%d: 0x%x\n", (i-traversal_stack), *i);
    }
}