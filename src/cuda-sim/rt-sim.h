#ifndef RAY_TRACE_FUNCTION_INCLUDED
#define RAY_TRACE_FUNCTION_INCLUDED


#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MIN_MAX(a,b,c) MAX(MIN((a), (b)), (c))
#define MAX_MIN(a,b,c) MIN(MAX((a), (b)), (c))

#define EMPTY_STACK 0x7654321
#define MAX_TRAVERSAL_STACK_SIZE 32

// #define DEBUG_PRINT

// Debugging
void print_float4(float4 printVal);
void print_stack(std::list<addr_t> &traversal_stack);

// Trace Ray
void trace_ray(const class ptx_instruction * pI, class ptx_thread_info * thread, const class function_info * target_func);
void trace_cwbvh(const class ptx_instruction * pI, class ptx_thread_info * thread, const class function_info * target_func, std::list<addr_t> & memory_accesses);
float magic_max7(float a0, float a1, float b0, float b1, float c0, float c1, float d);
float magic_min7(float a0, float a1, float b0, float b1, float c0, float c1, float d);
float3 get_t_bound(float3 box, float3 origin, float3 idirection);

// Predictor
uint32_t hash_comp(float x, uint32_t num_bits);
unsigned long long compute_hash(Ray ray, const float3& world_min, const float3& world_max);

// Ray Data Types
struct Hit
{
	float4 t_triId_u_v;
	//float4 triNormal_instId;
	//float4 geom_normal;
	//float4 texcoord;
};

struct meta
{
	unsigned char upper : 3, lower : 5;
};

struct CWBVHNode
{
	float3 pOrigin;
	char3 e;
	unsigned char imask;
	unsigned int nodeBaseIndex, triBaseIndex;
	uchar3 qlo[8];	
	uchar3 qhi[8];	
	meta childMetaData[8];

};


bool ray_box_test(float3 low, float3 high, float3 direction, float3 origin, float tmin, float tmax, float& thit);
bool ray_box_test_cwbvh(float3 low, float3 high, float3 idir, float3 origin, float tmin, float tmax, float& thit);
bool mt_ray_triangle_test(float3 p0, float3 p1, float3 p2, Ray ray_properties, float* thit);
bool rtao_ray_triangle_test(float4 v00, float4 v11, float4 v22, Ray ray_properties, float* thit, Hit& ray_payload);
bool rtao_ray_triangle_test(float4 v00, float4 v11, float4 v22, Ray ray_properties, float* thit);
Hit traverse_intersect(addr_t next_node, Ray ray_properties, addr_t node_start, addr_t tri_start, class ptx_thread_info * thread, memory_space * mem, std::map<new_addr_type, unsigned> &tree_level_map);

unsigned bfind(unsigned a);
unsigned popc(unsigned a);
float uint_as_float(unsigned int a);
float3 calculate_idir(float3 direction);

#endif