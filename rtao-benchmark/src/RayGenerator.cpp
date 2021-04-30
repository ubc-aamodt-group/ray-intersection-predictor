#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <math.h>
#include <ctime>
#include <assert.h>
#include <algorithm>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

#include "RayGenerator.h"
#include "OBJ_Loader.h"
#include "ValidationKernels.h"
#include "CUDAAssert.h"
#include "helper_math.h"

static bool compareByDirection(const Ray &a, const Ray &b)
{
  float4 a_d = a.dir_tmax;
  float4 b_d = b.dir_tmax;

  double a_d_d = a_d.x * 1/sqrt(3) + a_d.y * 1/sqrt(3) + a_d.z * 1/sqrt(3);
  a_d_d = (a_d_d < -1.0 ? -1.0 : (a_d_d > 1.0 ? 1.0 : a_d_d));
  
  double b_d_d = b_d.x * 1/sqrt(3) + b_d.y * 1/sqrt(3) + b_d.z * 1/sqrt(3);
  b_d_d = (b_d_d < -1.0 ? -1.0 : (b_d_d > 1.0 ? 1.0 : b_d_d));

  double a_angle = acos(a_d_d);
  double b_angle = acos(b_d_d);

  return a_angle < b_angle;
}

static bool compareByOrigin(const Ray &a, const Ray &b)
{
  float4 a_o = a.origin_tmin;
  float4 b_o = b.origin_tmin;

  float a_o_d = sqrt(a_o.x * a_o.x + a_o.y * a_o.y + a_o.z * a_o.z); // distance to origin
  float b_o_d = sqrt(b_o.x * b_o.x + b_o.y * b_o.y + b_o.z * b_o.z); // distance to origin

  return a_o_d < b_o_d;
}

RayGenerator::RayGenerator(const Camera& camera, uint p_spp, uint p_spt, float p_t_min, float p_t_max)
  : camera(camera), mt(std::random_device{}()), dist(0.0, 1.0)
{
  srand(time(0));
  
  spp = p_spp;
  samples_per_triangle = p_spt;
  t_min = p_t_min;
  t_max = p_t_max;

  pixelTable.setSize({ (int) camera.get_width(), (int) camera.get_height() });

  return;
}

RayGenerator::~RayGenerator()
{
  return;
}

int RayGenerator::loadModelOBJ(const std::string& model_path)
{

  objl::Loader loader;
  std::cout << model_path << std::endl;
  assert(loader.LoadFile(model_path));

  // loader.LoadFile("C:/Users/f.demoullin/Documents/graphics_assets/teapot.obj");
  // loader.LoadFile("C:/Users/f.demoullin/Documents/graphics_assets/conferenceRoomModel/conference.obj");
  // loader.LoadFile("C:/Users/f.demoullin/Documents/graphics_assets/AmazonLumberjard/exterior.obj");

  // generate Vertex and Index Buffers from obj loader:
  for (int i = 0; i < loader.LoadedIndices.size(); i += 3) 
  {
    int3 cur_index;
    cur_index.x = loader.LoadedIndices[i];
    cur_index.y = loader.LoadedIndices[i+1];
    cur_index.z = loader.LoadedIndices[i+2];
    IndexBuffer.push_back(cur_index);
  }
  for (int i = 0; i < loader.LoadedVertices.size(); i++) 
  {
    float3 cur_vertex;
    cur_vertex.x = loader.LoadedVertices[i].Position.X;
    cur_vertex.y = loader.LoadedVertices[i].Position.Y;
    cur_vertex.z = loader.LoadedVertices[i].Position.Z;
    VertexBuffer.push_back(cur_vertex);

    world_bounds_min = fminf(cur_vertex, world_bounds_min);
    world_bounds_max = fmaxf(cur_vertex, world_bounds_max);

    float3 cur_normal;
    cur_normal.x = loader.LoadedVertices[i].Normal.X;
    cur_normal.y = loader.LoadedVertices[i].Normal.Y;
    cur_normal.z = loader.LoadedVertices[i].Normal.Z;
    float length_cur_normal = sqrt(cur_normal.x*cur_normal.x + cur_normal.y*cur_normal.y + cur_normal.z*cur_normal.z);
    NormalBuffer.push_back(invertNormal * cur_normal/length_cur_normal); // HACK HACK HACK
  }

  assert(IndexBuffer.size() > 0);
  return(IndexBuffer.size());
}

// see pixar paper!
// Building an orthonormal basis revisited
void RayGenerator::GetTangentBasis(const float3 &n, float3 &b1, float3& b2)
{
  float sign = copysignf(1.0f, n.z);
  const float a = -1.0f / (sign + n.z);
  const float b = n.x * n.y * a;

  b1.x = 1.f + sign * n.x * n.x * a;
  b1.y = sign * b;
  b1.z = -sign * n.x;
  
  b2.x = b;
  b2.y = sign + n.y * n.y * a;
  b2.z = -n.y;
}

float3 RayGenerator::CosineSampleHemisphere()
{
  // generate random sample
  float2 E;
  E.x = dist(mt);
  E.y = dist(mt);
  
  // do the cosing hemisphere sampling!
  float Phi = 2 * M_PI * E.x;
  float CosTheta = sqrt(E.y);
  float SinTheta = sqrt(1 - CosTheta * CosTheta);

  float3 H = make_float3(SinTheta * cos(Phi), SinTheta * sin(Phi), CosTheta);
  return H;
}

void RayGenerator::generateSPPRaysFromWorldPosAndDir(float4 world_pos, float4 world_normal, std::vector<Ray>& out_rays, int my_spp)
{ 
  // option to change the spp count
  if (my_spp == 0)
  {
    my_spp = spp;
  }

  float3 n, b1, b2;

  const float world_normal_magnitude = sqrt(world_normal.x * world_normal.x + world_normal.y * world_normal.y + world_normal.z * world_normal.z);
  n.x = world_normal.x / world_normal_magnitude;
  n.y = world_normal.y / world_normal_magnitude;
  n.z = world_normal.z / world_normal_magnitude;

  // get orthonormal basis of the normal
  GetTangentBasis(n, b1, b2);
  
  for(int i=0; i < my_spp; i++)
  {
    // sample the hemisphere, get a tangent
    float3 tangent = CosineSampleHemisphere();

    // translate tangent into world_normal space
    // tangent * float3x3(row0 = b1, row1 = b2, row2 = n)
    float4 ray_direction;
    ray_direction.x = tangent.x * b1.x + tangent.y * b2.x + tangent.z * n.x;
    ray_direction.y = tangent.x * b1.y + tangent.y * b2.y + tangent.z * n.y;
    ray_direction.z = tangent.x * b1.z + tangent.y * b2.z + tangent.z * n.z;

    // normalize ray_direction - should not be required?
    ray_direction = ray_direction / sqrt(ray_direction.x * ray_direction.x + ray_direction.y * ray_direction.y + ray_direction.z * ray_direction.z);

    // encode for Ray struct
    ray_direction.w = t_max;
    world_pos.w = t_min;

    Ray cur_ray;
    cur_ray.origin_tmin = world_pos;
    cur_ray.dir_tmax = ray_direction;

    out_rays.push_back(cur_ray); 
  }
}

void RayGenerator::generateAORaysFromFile()
{
  std::string line;
  std::ifstream infile("C:/Users/f.demoullin/Documents/graphics_assets/ray_file_teapot.txt");
  // std::ifstream infile("C:/Users/f.demoullin/Documents/graphics_assets/ray_file_conference.txt");

  while (std::getline(infile, line))
  {
    std::stringstream iss(line);
    uint2 cur_coords;
    float4 world_pos;
    float4 world_normal;

    iss >> cur_coords.x;
    iss >> cur_coords.y;

    iss >> world_pos.x;
    iss >> world_pos.y;
    iss >> world_pos.z;

    iss >> world_normal.x;
    iss >> world_normal.y;
    iss >> world_normal.z;

    std::vector<Ray> temp;
    generateSPPRaysFromWorldPosAndDir(world_pos, world_normal, temp);
    ray_helper_vec.insert(std::end(ray_helper_vec), std::begin(temp), std::end(temp));
  }

  
}

/*
* See: https://www.cs.princeton.edu/~funk/tog02.pdf, Chapter 4.2
* For the proof, start here: https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle
*/
void RayGenerator::generatePointInsideTriangle(
  const float3 a, const float3 b, const float3 c,
  const float3 n_a, const float3 n_b, const float3 n_c, 
  float3& out_point, float3& out_normal)
  {     
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    float r1 = dist(mt);
    float r2 = dist(mt);

    out_point = (1 - sqrt(r1)) * a + (sqrt(r1) * (1 - r2)) * b + (sqrt(r1) * r2) * c;
    // assumption: the same linear combination can be used for the normal at out_point
    out_normal = (1 - sqrt(r1)) * n_a + (sqrt(r1) * (1 - r2)) * n_b + (sqrt(r1) * r2) * n_c;
  }

void RayGenerator::generatePrimaryRays(uint numSpp) {
  uint w = camera.get_width();
  uint h = camera.get_height();
  uint numPixels = w * h;

  ray_helper_vec.resize(numPixels * numSpp);

  // Generate rays in morton order for better coherence
  for (uint s = 0; s < numSpp; ++s) {
    tbb::parallel_for(0U, numPixels, [&](uint i) {
      uint pixel_index = pixelTable.getIndexToPixel()[i];
      uint x = pixel_index % w;
      uint y = pixel_index / w;

      // If more than one sample, randomly jitter to get anti-aliasing
      x += (numSpp > 1) ? dist(mt) : 0.5;
      y += (numSpp > 1) ? dist(mt) : 0.5;

      uint index = s * numPixels + i;
      Ray ray = camera.create_primary_ray(x, y, t_min, t_max);
      ray.path_index = i;
      ray_helper_vec[index] = ray;
    });
  }
  
  std::cout << std::dec;
  std::cout << w << "x" << h << "x" << numSpp << "=" << ray_helper_vec.size() << " primary rays generated" << std::endl;
}

void RayGenerator::processHits(Hit* hitResults, int numHits) {
  std::vector<Hit> hostHits(numHits);
  cudaCheck(cudaMemcpy(hostHits.data(), hitResults, sizeof(Hit) * numHits, cudaMemcpyDeviceToHost));

  // Interpolate attribute using barycentric coordinates
  const auto get_vertex_attribute = [](
    const std::vector<float3>& buffer, const int3& vertices, const float3& barycentric) {
    float3 a = buffer[vertices.x];
    float3 b = buffer[vertices.y];
    float3 c = buffer[vertices.z];
    return barycentric.x * a + barycentric.y * b + barycentric.z * c;
  };

  normal_vec.clear();
  normal_vec.reserve(hostHits.size());

  points_vec.clear();
  points_vec.reserve(hostHits.size());

  std::vector<Ray> temp_rays;
  temp_rays.reserve(hostHits.size());

  for (uint i = 0; i < hostHits.size(); i++) {
    const Hit& hit = hostHits[i];
    int id = *(int*) &hit.t_triId_u_v.y;

    // Miss
    if (id == -1) {
      continue;
    }

    float u = hit.t_triId_u_v.z;
    float v = hit.t_triId_u_v.w;
    float w = 1.0f - u - v;
    float3 barycentric = { u, v, w };

    int3 vertices = IndexBuffer[primitiveTriangleMap[id]];

    float3 normal = normalize(get_vertex_attribute(NormalBuffer, vertices, barycentric));
    float3 position = get_vertex_attribute(VertexBuffer, vertices, barycentric);

    normal_vec.push_back(normal);
    points_vec.push_back(position);
    temp_rays.push_back(ray_helper_vec[i]);
  }

  ray_helper_vec = std::move(temp_rays);
}

void RayGenerator::processHitStatus(Hit* hitResults, int numHits) {
  std::vector<Hit> hostHits(numHits);
  cudaCheck(cudaMemcpy(hostHits.data(), hitResults, sizeof(Hit) * numHits, cudaMemcpyDeviceToHost));

  ray_hit_status.clear();
  ray_hit_status.reserve(hostHits.size());

  for (uint i = 0; i < hostHits.size(); i++) {
    const Hit& hit = hostHits[i];
    int id = *(int*) &hit.t_triId_u_v.y;

    ray_hit_status.push_back(id != -1);
  }
}

void RayGenerator::generateSecondaryRays(uint numRaysPerHit) {
  std::vector<Ray> temp_rays;
  temp_rays.resize(ray_helper_vec.size() * numRaysPerHit);

  tbb::parallel_for(0U, (uint) ray_helper_vec.size(), [&](uint i) {
    float3 normal = normal_vec[i];
    float3 position = points_vec[i];
    float3 b1, b2;
    GetTangentBasis(normal, b1, b2);

    for (uint j = 0; j < numRaysPerHit; j++) {
      // Cosine sample upper hemisphere for new ray direction
      float3 tangent = CosineSampleHemisphere();
      float3 direction;
      direction.x = tangent.x * b1.x + tangent.y * b2.x + tangent.z * normal.x;
      direction.y = tangent.x * b1.y + tangent.y * b2.y + tangent.z * normal.y;
      direction.z = tangent.x * b1.z + tangent.y * b2.z + tangent.z * normal.z;
      direction = normalize(direction);

      Ray ray;
      ray.make_ray(
          position.x, position.y, position.z,
          direction.x, direction.y, direction.z,
          t_min, t_max);
      ray.path_index = ray_helper_vec[i].path_index;

      temp_rays[i * numRaysPerHit + j] = ray;
    }
  });

  ray_helper_vec = std::move(temp_rays);
}

void RayGenerator::generateReflectionRays() {
  generateSecondaryRays(1);

  std::cout << std::dec;
  std::cout << ray_helper_vec.size() << " reflection rays generated" << std::endl;
}

void RayGenerator::generateAORays() {
  generateSecondaryRays(spp);

  std::cout << std::dec;
  std::cout << ray_helper_vec.size() << " AO rays generated" << std::endl;
}

uint64_t hash_float(float x, uint32_t num_bits) {
  uint32_t mask = UINT32_MAX >> (32 - num_bits);

  uint32_t o_x = *((uint32_t*) &x);

  uint64_t sign_bit_x = o_x >> 31;
  uint64_t exp_x = (o_x >> (31 - num_bits)) & mask;
  uint64_t mant_x = (o_x >> (23 - num_bits)) & mask;

  return (sign_bit_x << (2 * num_bits)) | (exp_x << num_bits) | mant_x;
}

// Quantize direction to a sphere
uint64_t hash_direction_spherical(const float4 &d, uint32_t theta_bits) {
  uint32_t phi_bits = theta_bits + 1;

  uint64_t theta = std::acos(clamp(d.z, -1.f, 1.f)) / Pi * 180;
  uint64_t phi = (std::atan2(d.y, d.x) + Pi) / Pi * 180;
  uint64_t q_theta = theta >> (8 - theta_bits);
  uint64_t q_phi = phi >> (9 - phi_bits);

  return (q_phi << theta_bits) | q_theta;
}

// Quantize origin to a grid
uint64_t hash_origin_grid(const float4 &o, const float3& min,
                          const float3& max, uint32_t num_bits) {
  uint32_t grid_size = 1 << num_bits;

  uint64_t hash_o_x = clamp((o.x - min.x) / (max.x - min.x) * grid_size, 0.f, (float)grid_size - 1);
  uint64_t hash_o_y = clamp((o.y - min.y) / (max.y - min.y) * grid_size, 0.f, (float)grid_size - 1);
  uint64_t hash_o_z = clamp((o.z - min.z) / (max.z - min.z) * grid_size, 0.f, (float)grid_size - 1);
  return (hash_o_x << (2 * num_bits)) | (hash_o_y << num_bits) | hash_o_z;
}

// Francois' predictor hash function
uint64_t RayGenerator::hash_francois(const Ray &ray, uint32_t num_bits)
{
  uint32_t num_bits_hash = 2 * num_bits + 1;

  uint64_t hash_d =
    (hash_float(ray.dir_tmax.z, num_bits) << (2 * num_bits_hash)) |
    (hash_float(ray.dir_tmax.y, num_bits) << num_bits_hash) |
     hash_float(ray.dir_tmax.x, num_bits);
  uint64_t hash_o =
    (hash_float(ray.origin_tmin.x, num_bits) << (2 * num_bits_hash)) |
    (hash_float(ray.origin_tmin.y, num_bits) << num_bits_hash) |
     hash_float(ray.origin_tmin.z, num_bits);
  uint64_t hash = hash_o ^ hash_d;

  return hash;
}

uint64_t RayGenerator::hash_francois_grid_spherical(const Ray &ray, const float3& min,
                                                    const float3& max,
                                                    uint32_t num_francois_bits,
                                                    uint32_t num_grid_bits,
                                                    uint32_t num_theta_bits) {
  uint64_t hash_d = hash_direction_spherical(ray.dir_tmax, num_theta_bits);
  uint64_t hash_o = hash_origin_grid(ray.origin_tmin, min, max, num_grid_bits);
  return hash_o ^ hash_d ^ hash_francois(ray, num_francois_bits);
}

uint64_t RayGenerator::hash_grid_spherical(const Ray &ray)
{
  uint64_t hash_d = hash_direction_spherical(ray.dir_tmax, 3);
  uint64_t hash_o = hash_origin_grid(ray.origin_tmin, world_bounds_min, world_bounds_max, 5);
  uint64_t hash = (hash_o << 7) | hash_d;

  return hash;
}

void RayGenerator::raySorting(const ray_sorting sorting_strategy)
{
  if (sorting_strategy == no_sort) {
    return;
  }

  printf("Sorting rays...\n");

  switch(sorting_strategy)
  {
    case random_shuffle: 
    {
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(ray_helper_vec.begin(), ray_helper_vec.end(), g);
      break;
    }
    case direction:
    {
      sort(ray_helper_vec.begin(), ray_helper_vec.end(), compareByDirection);
      break;
    }
    case origin:
    {
      sort(ray_helper_vec.begin(), ray_helper_vec.end(), compareByOrigin);
      break;
    }
    case origin_chunk:
    {
      int chunk_size = 8192;
      int cur_index = 0;
      
      while (cur_index < ray_helper_vec.size())
      {
        auto begin_it = ray_helper_vec.begin() + cur_index;
        auto end_it = ray_helper_vec.begin() + cur_index + chunk_size;
        end_it = min(end_it, ray_helper_vec.end());

        sort(begin_it, end_it, compareByOrigin);
        cur_index += chunk_size;
      }
      break;
    }
    case hash:
    {
      tbb::parallel_sort(ray_helper_vec.begin(), ray_helper_vec.end(),
        [&](const Ray& a, const Ray& b) {
          return hash_grid_spherical(a) < hash_grid_spherical(b);
        });
      break;
    }
    default:
    {
      assert(false);
    }
  }
}

/*
Generates points and normals. 
Generate 1 point for each traingle on model, repeat spt times
*/
int RayGenerator::generatePointsAndNormals_model(int number_of_rays)
{
  int sample_count = 0;

  for (int cur_samples_per_triangle = 0; cur_samples_per_triangle < samples_per_triangle; cur_samples_per_triangle++)
  {
    for (auto cur_index : IndexBuffer)
    {
      float3 cur_point, cur_normal;
      generatePointInsideTriangle(
        VertexBuffer[cur_index.x], VertexBuffer[cur_index.y], VertexBuffer[cur_index.z],
        NormalBuffer[cur_index.x], NormalBuffer[cur_index.y], NormalBuffer[cur_index.z],
        cur_point, cur_normal);

      points_vec.push_back(cur_point);
      normal_vec.push_back(cur_normal);

      sample_count++;
    }
    if (number_of_rays > 0 && sample_count > number_of_rays) break;
  }

  return sample_count;
}

/*
Generates points and normals. 
Generate spt points for each triangle
*/
int RayGenerator::generatePointsAndNormals_spt(int number_of_rays)
{
  int sample_count = 0;

  for (auto cur_index : IndexBuffer)
  {
    for (int cur_samples_per_triangle = 0; cur_samples_per_triangle < samples_per_triangle; cur_samples_per_triangle++)
    {
      float3 cur_point, cur_normal;
      generatePointInsideTriangle(
        VertexBuffer[cur_index.x], VertexBuffer[cur_index.y], VertexBuffer[cur_index.z],
        NormalBuffer[cur_index.x], NormalBuffer[cur_index.y], NormalBuffer[cur_index.z],
        cur_point, cur_normal);

      points_vec.push_back(cur_point);
      normal_vec.push_back(cur_normal);

      sample_count++;
    }
    if (number_of_rays > 0 && sample_count > number_of_rays) break;
  }

  return sample_count;
}

int RayGenerator::generate_detangled_spp()
{ 
  assert(points_vec.size() == normal_vec.size());
  assert(points_vec.size() > 0 && normal_vec.size() > 0);
  assert(ray_helper_vec.size() == 0);

  for (int i=0; i < points_vec.size(); i++)
  {
    float3 cur_point = points_vec[i];
    float3 cur_normal = normal_vec[i];

    std::vector<Ray> temp;
    generateSPPRaysFromWorldPosAndDir(make_float4(cur_point), make_float4(cur_normal), temp);
    ray_helper_vec.insert(std::end(ray_helper_vec), std::begin(temp), std::end(temp));
  }
  return ray_helper_vec.size();
}

int RayGenerator::generate_entangled_spp(int entangle_size)
{ 
  assert(points_vec.size() == normal_vec.size());
  assert(points_vec.size() > 0 && normal_vec.size() > 0);
  assert(ray_helper_vec.size() == 0);

  for (int my_spp=0; my_spp < spp; my_spp++)
  {
    for (int j=0; j < points_vec.size(); j+= entangle_size)
    {
      float3 cur_point = points_vec[j];
      float3 cur_normal = normal_vec[j];
      
      std::vector<Ray> temp;
      generateSPPRaysFromWorldPosAndDir(make_float4(cur_point), make_float4(cur_normal), temp, 1); 
      assert(temp.size() == 1);

      Ray reference_ray = temp[0];
      float4 reference_direction = reference_ray.get_direction();

      for (int i=1; i < entangle_size; i++)
      {
        cur_point = points_vec[j+i];
        cur_normal = normal_vec[j+i];

        Ray r;
        r.make_ray(cur_point.x, cur_point.y, cur_point.z, reference_direction.x, reference_direction.y, reference_direction.z, t_min, t_max);
        temp.push_back(r);
      }

      assert(temp.size() == entangle_size);
      ray_helper_vec.insert(std::end(ray_helper_vec), std::begin(temp), std::end(temp));
    }
  }  
  return ray_helper_vec.size();
}

void RayGenerator::split_warmup_measurement_rays(uint num_warmup_rays) {
  if (num_warmup_rays > 0) {
    ray_helper_vec_measurement =
      std::vector<Ray>(ray_helper_vec.begin() + num_warmup_rays, ray_helper_vec.end());
    ray_helper_vec =
      std::vector<Ray>(ray_helper_vec.begin(), ray_helper_vec.begin() + num_warmup_rays);
    std::cout << std::dec
              << "Split into " << ray_helper_vec.size() << " warmup rays and "
              << ray_helper_vec_measurement.size() << " measurement rays" << std::endl;
  }
}
void RayGenerator::load_measurement_rays() {
  std::cout << std::dec
            << ray_helper_vec_measurement.size() << " measurement rays loaded" << std::endl;
  ray_helper_vec = ray_helper_vec_measurement;
  ray_helper_vec_measurement.clear();
}

void RayGenerator::uploadRaysToGPU()
{
  cudaCheck(cudaFree(cudaRays));
  cudaRays = GenerateRaysFromFile(ray_helper_vec, ray_helper_vec.size());
  unsigned long cuda_rays_size = sizeof(Ray) * ray_helper_vec.size();
  print_helper::print_buffer("cudaRays", cuda_rays_size, (void*)cudaRays);
}

void RayGenerator::uploadSingleRayToGPU(int ray_id)
{
  std::vector<Ray> helper_vec;
  helper_vec.push_back(ray_helper_vec[ray_id]);

  assert(helper_vec.size() == 1);
  
  cudaRays = GenerateRaysFromFile(helper_vec, helper_vec.size());
  unsigned long cuda_rays_size = sizeof(Ray) * helper_vec.size();
  print_helper::print_buffer("cudaRays", cuda_rays_size, (void*)cudaRays);
}

void RayGenerator::saveRaysToFile(const std::string& file_path, const std::string& model_name)
{
  std::ofstream myfile;

  // make meaningful filename: (model + number_of_rays + date)
  time_t t = time(NULL);
  tm* timePtr = localtime(&t);

  const std::string file_name = file_path 
    + model_name
    + "_" + std::to_string(ray_helper_vec.size()) 
    + "_" + std::to_string(timePtr->tm_mday) 
    + "_" + std::to_string(timePtr->tm_mon)
    + ".ray_file";

  std::cout << file_name << std::endl;

  myfile.open(file_name);

  for (auto cur_ray : ray_helper_vec)
  {

    myfile 
      << cur_ray.origin_tmin.x << " "
      << cur_ray.origin_tmin.y << " "
      << cur_ray.origin_tmin.z << " "
      << cur_ray.dir_tmax.x << " "
      << cur_ray.dir_tmax.y << " "
      << cur_ray.dir_tmax.z << " "
      << cur_ray.origin_tmin.w << " "
      << cur_ray.dir_tmax.w << " "
      << std::endl;
  }
  myfile.close();
}

void RayGenerator::readRaysFromFile(const std::string& file_path, const uint number_of_rays)
{
  std::ifstream file(file_path);

  if (!file.is_open()) {
    throw std::runtime_error("Unable to open " + file_path);
  }
  
  std::string line;

  int i = 0;
  ray_helper_vec.clear();

  while(std::getline(file, line) && i < number_of_rays)
  {
    if (line.empty()) {
      break;
    }

    std::vector<float> lineData;
    std::stringstream lineStream(line);
    float value;

    while(lineStream >> value)
    {
      lineData.push_back(value);
    }
    
    assert(lineData.size() == 8);
    ray_helper_vec.emplace_back();
    ray_helper_vec.back().make_ray(lineData[0], lineData[1], lineData[2], lineData[3], lineData[4], lineData[5], lineData[6], lineData[7]); 
    i++;
  }

  std::cout << std::dec << ray_helper_vec.size() << " rays loaded" << std::endl;
}

void RayGenerator::fillWithListOfRays()
{
  ray_helper_vec.resize(28);
  int i = 0; 
  
  ray_helper_vec[i].make_ray(3.568816, 2.421769, 1.685847, 0.032519, 0.699330, -0.714059, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(-0.564556, 0.512781, 0.923323, -0.677293, -0.557728, -0.479807, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(-0.876105, 0.252556, 0.895642, 0.509877, -0.070725, -0.857335, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(-5.327621, 3.557689, -0.173771, -0.841987, 0.371578, 0.391138, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(-1.959287, 8.296237, -2.353784, -0.261663, 0.953761, -0.147895, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(3.222832, 1.693200, -2.044262, -0.134033, -0.101980, -0.985716, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(-3.591162, 6.289504, -1.105651, -0.288273, 0.503503, 0.814483, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(0.890832, 0.597294, 0.636748, -0.964852, -0.252661, 0.072264, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(3.737526, 1.940752, -1.444702, 0.177436, 0.944807, 0.275420, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(-4.481230, 6.163420, -0.183426, 0.786466, -0.616637, 0.035079, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(1.808422, 1.377815, -2.108364, 0.364215, -0.290656, -0.884798, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(-4.308555, 6.676151, -1.043637, -0.971864, -0.230413, -0.048899, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(-1.034808, 1.331691, 1.351067, -0.623053, 0.765845, -0.159016, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(2.852999, 4.354635, 0.766796, -0.555939, -0.462622, -0.690588, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(-0.127557, 2.460680, -1.834043, 0.427201, 0.679930, -0.595982, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(3.153050, 1.491750, 1.507839, 0.558117, -0.823757, 0.099647, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(-1.584222, 0.055246, -1.928574, 0.195063, -0.685890, 0.701074, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(-6.348334, 6.920264, 0.596720, -0.561330, 0.825069, 0.064580, 0.100000, 10.000000); i++; // PROBLEM RAY
  ray_helper_vec[i].make_ray(0.811605, 2.763596, -0.587691, 0.653653, 0.646147, -0.393995, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(2.864032, 1.010957, 1.483948, -0.140409, 0.392348, 0.909037, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(-3.819824, 5.321078, -0.781778, 0.144730, 0.781929, -0.606334, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(4.852672, 3.102771, 0.048357, -0.227542, -0.275382, -0.934018, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(3.369248, 1.893227, 1.567559, -0.034272, -0.862339, -0.505170, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(1.559892, 2.120020, -0.234336, 0.159782, -0.192666, 0.968168, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(3.095580, 2.972372, 1.332161, 0.407788, 0.589411, -0.697355, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(-5.363301, 0.459256, -1.351863, 0.229949, -0.477988, -0.847733, 0.100000, 10.000000); i++; // PROBLEM RAY
  ray_helper_vec[i].make_ray(-0.598682, 5.248319, -0.568175, -0.692829, 0.694979, 0.192336, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(-1.212672, 6.555009, -0.406828, 0.396314, -0.697342, -0.597202, 0.100000, 10.000000); i++;
  /*
  ray_helper_vec[i].make_ray(2.895570, 1.460072, 1.266253, 0.829048, 0.447160, -0.335749, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(-3.820758, 5.341610, -0.177821, -0.702005, 0.691220, 0.171475, 0.100000, 10.000000); i++;
  ray_helper_vec[i].make_ray(6.188961, 5.243276, -0.424368, 0.489324, 0.222677, 0.843195, 0.100000, 10.000000); i++;
  
  ray_helper_vec[i].make_ray(3.030370, 1.770496, 1.242463, 0.550894, -0.726202, -0.411274, 0.100000, 10.000000); i++;
  */

  assert(i == ray_helper_vec.size()); 

  /*
  for( int i = 0; i < ray_helper_vec.size(); i++)
  {
    printf("ray_helper_vec[i].make_ray(%f, %f, %f, %f, %f, %f, %f, %f); i++;\n",
      ray_helper_vec[i].origin_tmin.x, ray_helper_vec[i].origin_tmin.y, ray_helper_vec[i].origin_tmin.z,
      ray_helper_vec[i].dir_tmax.x, ray_helper_vec[i].dir_tmax.y, ray_helper_vec[i].dir_tmax.z,
      ray_helper_vec[i].origin_tmin.w, ray_helper_vec[i].dir_tmax.w
    );
  }
  */
}