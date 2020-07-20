#ifndef RAY_TRACE_FUNCTION_INCLUDED
#define RAY_TRACE_FUNCTION_INCLUDED


#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MIN_MAX(a,b,c) MAX(MIN((a), (b)), (c))
#define MAX_MIN(a,b,c) MIN(MAX((a), (b)), (c))

#define EMPTY_STACK 0x7654321
#define MAX_TRAVERSAL_STACK_SIZE 32

// Debugging
void print_float4(float4 printVal);
void print_stack(std::list<addr_t> &traversal_stack);

// Trace Ray
void trace_ray(const class ptx_instruction * pI, class ptx_thread_info * thread, const class function_info * target_func, std::list<addr_t> & memory_accesses);
float magic_max7(float a0, float a1, float b0, float b1, float c0, float c1, float d);
float magic_min7(float a0, float a1, float b0, float b1, float c0, float c1, float d);
float3 get_t_bound(float3 box, float3 origin, float3 direction);

// Ray Data Types
struct Ray
{
	float4 origin_tmin;
	float4 dir_tmax;

  float3 get_origin() { return {origin_tmin.x, origin_tmin.y, origin_tmin.z}; }
  void set_origin(float3 new_origin) { origin_tmin = {new_origin.x, new_origin.y, new_origin.z, origin_tmin.w}; }
  
  float get_tmin() { return origin_tmin.w; }
  float get_tmax() { return dir_tmax.w; }

  float3 get_direction() { return {dir_tmax.x, dir_tmax.y, dir_tmax.z}; }
  void set_direction(float4 new_dir) { dir_tmax = new_dir; }
};

struct Hit
{
	float4 t_triId_u_v;
	//float4 triNormal_instId;
	//float4 geom_normal;
	//float4 texcoord;
};


bool ray_box_test(float3 low, float3 high, float3 inv_dir, float3 origin, float tmin, float tmax, float& thit);
bool mt_ray_triangle_test(float3 p0, float3 p1, float3 p2, Ray ray_properties, float* thit);
bool rtao_ray_triangle_test(float4 v00, float4 v11, float4 v22, Ray ray_properties, float* thit, Hit& ray_payload);



#endif