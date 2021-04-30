#include "helper_math.h"
#include "Camera.h"
#include "PixelTable.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <random>

class BVHManager;

class RayGenerator
{
  public:
  
  RayGenerator(const Camera& camera, uint p_spp=1, uint p_spt=1, float p_t_min=0.1, float p_t_max=10);
  ~RayGenerator();

  // populates VertexBuffer and IndexBuffer
  int loadModelOBJ(const std::string& model_path);

  // generate ray_helper_vec and cudaRays - chose either one!
  void generateAORaysFromFile();
  void generatePointsAndNormals(int number_of_rays=-1);

  /* Generates points and normals */
  // Generate 1 point for each traingle on model, repeat spt times
  int generatePointsAndNormals_model(int number_of_rays=-1);
  // Generate spt points for each triangle
  int generatePointsAndNormals_spt(int number_of_rays=-1);

  void generatePrimaryRays(uint numSpp);
  void generateSecondaryRays(uint numRaysPerHit);
  void generateReflectionRays();
  void generateAORays();
  void processHits(Hit* hitResults, int numHits);
  void processHitStatus(Hit* hitResults, int numHits);

  void RandomizeAndDownsizeRays();
  void uploadRaysToGPU();
  void uploadSingleRayToGPU(int ray_id);
  
  int getRayCount() { return ray_helper_vec.size(); }

  int generate_entangled_spp(int entangle_size);
  int generate_detangled_spp();

  // ray file IO - this is used to make rays deterministic
  void saveRaysToFile(const std::string& file_path, const std::string& model_name);
  void readRaysFromFile(const std::string& file_path, const uint number_of_rays);

  enum ray_sorting { no_sort, random_shuffle, direction, origin, origin_chunk, hash };
  void raySorting(const ray_sorting sorting_strategy = m_ray_sorting_strategy);

  uint64_t hash_francois(const Ray &ray, uint32_t num_bits);
  uint64_t hash_francois_grid_spherical(const Ray &ray, const float3& min,
                                        const float3& max,
                                        uint32_t num_francois_bits,
                                        uint32_t num_grid_bits,
                                        uint32_t num_theta_bits);
  uint64_t hash_grid_spherical(const Ray &ray);

  void clear_rays() { ray_helper_vec.clear(); }
  void split_warmup_measurement_rays(uint num_warmup_rays);
  void load_measurement_rays();
  bool has_measurement_rays() const { return !ray_helper_vec_measurement.empty(); }

  // debugging
  void fillWithListOfRays();

  // Hack! Invert the normal for the Tepot model because of the OBJ file
  // const int invertNormal = -1; // teapot
  const int invertNormal = 1; // sponza + dragon + san-mig

  private:

  Camera camera;
  PixelTable pixelTable;

  float t_min;
  float t_max;
  uint spp;
  uint samples_per_triangle;

  std::vector<Ray> ray_helper_vec;
  std::vector<Ray> ray_helper_vec_measurement;
  std::vector<bool> ray_hit_status;
  Ray* cudaRays = nullptr;

  std::vector<float3> VertexBuffer;
  std::vector<float3> NormalBuffer;
  std::vector<int3> IndexBuffer;

  std::vector<float3> points_vec;
  std::vector<float3> normal_vec;

  std::unordered_map<int, int> primitiveTriangleMap;

  float3 world_bounds_min;
  float3 world_bounds_max;

  std::mt19937 mt;
  std::uniform_real_distribution<float> dist;

  // give rays the proper ray direction
  float3 CosineSampleHemisphere();
  void GetTangentBasis(const float3 &n, float3 &b1, float3& b2);
  void generateSPPRaysFromWorldPosAndDir(float4 world_pos, float4 world_normal, std::vector<Ray>& out_rays, int my_spp=0);
  void generatePointInsideTriangle(
    const float3 a, const float3 b, const float3 c,
    const float3 n_a, const float3 n_b, const float3 n_c, 
    float3& out_point, float3& out_normal);

  static const ray_sorting m_ray_sorting_strategy = direction;
  
  friend class BVHManager;
  friend class RayTraceManager;
};