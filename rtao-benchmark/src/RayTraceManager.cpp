
#include "RayTraceManager.h"
#include "RayGenerator.h"
#include "CUDAAssert.h"
#include "TraversalKernelBVH2.h"
#include "TraversalKernelCWBVH.h"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <stb_image/stb_image.h>
#include <stb_image/stb_image_write.h>

#ifdef OPTIX_PRIME
#include <optix_prime/optix_primepp.h>
#include <optix.h>
#endif

void RayTraceManager::allocCudaHits(size_t num) {
  cudaCheck(cudaFree(cudaHits));
  cudaCheck(cudaMalloc(&cudaHits, sizeof(Hit) * num));
  unsigned long cuda_hits_size = sizeof(Hit) * num;
  print_helper::print_buffer("cudaHits", cuda_hits_size, (void*)cudaHits);
}

RayTraceManager::RayTraceManager(RayGenerator& rg)
{
  numRays = rg.ray_helper_vec.size();
  anyhit = false;
  cudaHits = nullptr;

  if (numRays > 0)
    allocCudaHits(numRays);
};

void RayTraceManager::setNumRays(uint n)
{
  if (n > numRays) {
    allocCudaHits(n);
  }
  numRays = n;
}

#ifdef OPTIX_PRIME
void RayTraceManager::traceOptiXPrime(RayGenerator& rg)
{
  optix::prime::Context OptiXContext = optix::prime::Context::create(RTP_CONTEXT_TYPE_CUDA);
  optix::prime::Model SceneModel = OptiXContext->createModel();
  SceneModel->setTriangles(rg.IndexBuffer.size(), RTP_BUFFER_TYPE_HOST, rg.IndexBuffer.data(), rg.VertexBuffer.size(), RTP_BUFFER_TYPE_HOST, rg.VertexBuffer.data());
  SceneModel->update(RTP_MODEL_HINT_NONE);
  SceneModel->finish();

  optix::prime::Query query = SceneModel->createQuery(
    anyhit ? RTP_QUERY_TYPE_ANY : RTP_QUERY_TYPE_CLOSEST);
  query->setRays(numRays, RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX, RTP_BUFFER_TYPE_CUDA_LINEAR, rg.cudaRays);
  query->setHits(numRays, RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V, RTP_BUFFER_TYPE_CUDA_LINEAR, cudaHits);

  cudaProfilerStart();
  {
    float elapsedTime;
    cudaEvent_t startEvent, stopEvent;
    cudaCheck(cudaEventCreate(&startEvent));
    cudaCheck(cudaEventCreate(&stopEvent));
    cudaCheck(cudaEventRecord(startEvent, 0));
    query->execute(0);
    cudaCheck(cudaEventRecord(stopEvent, 0));
    cudaCheck(cudaEventSynchronize(stopEvent));
    cudaCheck(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

    Log("%.3fMS, %.2fMRays/s (OptiXPrime)", elapsedTime, (float)numRays / 1000000.0f / (elapsedTime / 1000.0f));
  }
  cudaProfilerStop();
}
#endif

void RayTraceManager::traceAila(RayGenerator& rg)
{
#ifdef SINGLERAY
  printf("DEBUG: Tracing 1 ray...\n");
  numRays = 1;
// #elif MAGIC
//   printf("MAGIC: Tracing 128 ray...\n");
//   numRays = 128;
#endif
  rtTraceBVH2(rg.cudaRays, cudaHits, numRays, anyhit);
}

void RayTraceManager::traceCWBVH(RayGenerator& rg)
{
  #ifdef SINGLERAY
  printf("DEBUG: Tracing 1 ray...\n");
  numRays = 1;
  #endif
  rtTraceCWBVH(rg.cudaRays, cudaHits, numRays, anyhit);
}

void RayTraceManager::traceCWBVHSingleRay(RayGenerator& rg)
{
  numRays = 1; // this is a hack!
  rtTraceCWBVH(rg.cudaRays, cudaHits, numRays, anyhit);
}

void RayTraceManager::traceOptiX(RayGenerator& rg)
{
  /*TODO!!*/
  // RTcontext context = 0;
  // RT_CHECK_ERROR(rtContextCreate(&context));
}

void RayTraceManager::pathtrace(RayGenerator& rg, TraceType type)
{
  void (RayTraceManager::*traceFunc)(RayGenerator&);

  switch (type)
  {
  case TraceType::Aila:
    traceFunc = &RayTraceManager::traceAila;
    break;
  case TraceType::CWBVH:
    traceFunc = &RayTraceManager::traceCWBVH;
    break;
#ifdef OPTIX_PRIME
  case TraceType::OptiXPrime:
    traceFunc = &RayTraceManager::traceOptiXPrime;
    break;
#endif
  default:
    throw std::invalid_argument("Unknown trace type");
  }

  uint w = rg.camera.get_width();
  uint h = rg.camera.get_height();
  uint numPixels = w * h;
  allocCudaHits(numPixels * rg.spp);
  setAnyHit(false);

  pixels.clear();
  pixels.resize(numPixels);

  // Begin render loop by generating and intersect primary rays
  rg.generatePrimaryRays(rg.spp);
  numRays = rg.getRayCount();

  rg.uploadRaysToGPU();
  (this->*traceFunc)(rg);

  rg.processHits(cudaHits, numRays);
  updatePixels(rg);

  // Generate and intersect reflection rays
  for (int depth = 0; depth < maxDepth; ++depth) {
    rg.generateReflectionRays();

    if (sort) {
      rg.raySorting(rg.hash);
    }
    
    numRays = rg.getRayCount();

    if (numRays == 0) {
      break;
    }

    rg.uploadRaysToGPU();
    (this->*traceFunc)(rg);

    rg.processHits(cudaHits, numRays);
    updatePixels(rg);
  }
}

void RayTraceManager::ao(RayGenerator& rg, TraceType type)
{
  void (RayTraceManager::*traceFunc)(RayGenerator&);

  switch (type)
  {
  case TraceType::Aila:
    traceFunc = &RayTraceManager::traceAila;
    break;
  case TraceType::CWBVH:
    traceFunc = &RayTraceManager::traceCWBVH;
    break;
#ifdef OPTIX_PRIME
  case TraceType::OptiXPrime:
    traceFunc = &RayTraceManager::traceOptiXPrime;
    break;
#endif
  default:
    throw std::invalid_argument("Unknown trace type");
  }

  uint w = rg.camera.get_width();
  uint h = rg.camera.get_height();
  uint numPixels = w * h;
  allocCudaHits(numPixels * rg.spp);

  pixels.clear();
  pixels.resize(numPixels);

  // Trace primary rays to get hits
  setAnyHit(false);
  rg.generatePrimaryRays(1);
  numRays = rg.getRayCount();

  rg.uploadRaysToGPU();
  (this->*traceFunc)(rg);

  rg.processHits(cudaHits, numRays);

  // Generate and intersect AO rays
  setAnyHit(true);
  rg.generateAORays();

  if (sort) {
    rg.raySorting(rg.hash);
  }

  numRays = rg.getRayCount();

  if (numRays == 0) {
    return;
  }

  rg.uploadRaysToGPU();
  (this->*traceFunc)(rg);

  rg.processHitStatus(cudaHits, numRays);
  updatePixels(rg);
}

void RayTraceManager::updatePixels(RayGenerator& rg) {
  std::vector<float> colors(pixels.size());

  if (shading == "ao") {
    for (uint i = 0; i < rg.ray_helper_vec.size(); ++i) {
      bool hit = rg.ray_hit_status[i];

      if (!hit) {
        uint path_index = rg.ray_helper_vec[i].path_index;
        uint pixel_index = rg.pixelTable.getIndexToPixel()[path_index];
        colors[pixel_index] += 1.0f;
      }
    }
  } else {
    for (uint i = 0; i < rg.ray_helper_vec.size(); ++i) {
      int path_index = rg.ray_helper_vec[i].path_index;
      
      float3 normal = rg.normal_vec[i];
      float3 point = rg.points_vec[i];

      // Assume lambertian surface, light at camera position
      uint pixel_index = rg.pixelTable.getIndexToPixel()[path_index];

      if (shading == "basic") {
        colors[pixel_index] += 1.0f;
      } else if (shading == "diffuse") {
        colors[pixel_index] += std::max(dot(normal, normalize(rg.camera.get_position() - point)), 0.0f);
      }
      colors[pixel_index] /= (maxDepth + 1);
    }
  }

  for (uint i = 0; i < colors.size(); ++i) {
    // Accumulate colour
    float color = colors[i] / rg.spp;
    pixels[i] += { color, color, color };
  }
}

void RayTraceManager::outputToImage(RayGenerator& rg, const std::string& out_image_path) {
  uint w = rg.camera.get_width();
  uint h = rg.camera.get_height();

  std::vector<uint8_t> pixel_buf(w * h * 3);

  const auto tonemap = [](const float3& x) { return x / (x + 1.0f); };
  const auto gamma_correct = [](const float3& x) -> float3 {
    return {
      powf(x.x, 1.0f / 2.2f),
      powf(x.y, 1.0f / 2.2f),
      powf(x.z, 1.0f / 2.2f),
    };
  };

  for (uint i = 0; i < pixels.size(); ++i) {
    float3 pixel = pixels[i];
    if (shading == "diffuse") {
      // Scale color to [0, 1] range
      pixel = tonemap(pixel);
    }
    pixel = gamma_correct(pixel);
    
    pixel_buf[i * 3] = pixel.x * 255;
    pixel_buf[i * 3 + 1] = pixel.y * 255;
    pixel_buf[i * 3 + 2] = pixel.z * 255;
  }

  int success = stbi_write_jpg(out_image_path.c_str(), w, h, STBI_rgb, pixel_buf.data(), 100);

  if (!success) {
    throw std::runtime_error("Failed to save image " + out_image_path);
  } else {
    std::cout << "Saved image to " << out_image_path << std::endl;
  }
}

void RayTraceManager::evaluateAndPrintForPLYVisualization(RayGenerator& rg, const std::string& out_ply_path)
{
  std::vector<Hit> hostHits(numRays);
  cudaCheck(cudaMemcpy(hostHits.data(), cudaHits, sizeof(Hit) * numRays, cudaMemcpyDeviceToHost));
  assert(numRays % rg.spp == 0);
  printf("Visualizer detected: %i spp\n", rg.spp);

  std::cout << out_ply_path << std::endl;

  std::ofstream outfile(out_ply_path);

  // header
  outfile << "ply" << std::endl;
  outfile << "format ascii 1.0" << std::endl;
  outfile << "element vertex " << numRays/rg.spp << std::endl;
  outfile << "property float x" << std::endl;
  outfile << "property float y" << std::endl;
  outfile << "property float z" << std::endl;
  outfile << "property uchar red" << std::endl;
  outfile << "property uchar green" << std::endl;
  outfile << "property uchar blue" << std::endl;
  outfile << "end_header" << std::endl;

  std::cout << "numRays: " << (int)numRays << " rg.spp: " << rg.spp << std::endl;
  // data (greyscale)
  for (int i = 0; i < numRays/rg.spp; i++)
  {
    float sum = 0.f;
    for (int j = 0; j < rg.spp; j++)
    {
      float cur_t = hostHits[i * rg.spp + j].t_triId_u_v.x;
      if (cur_t > rg.t_min && cur_t < rg.t_max)
      {
        assert(cur_t >= rg.t_min);
        assert(cur_t <= rg.t_max);
        sum += 1;
      }
    }

    if (sum > 0)
    {
      uint color = (uint)((255.f / (float)rg.spp) * sum);
      outfile
        << rg.ray_helper_vec[i * rg.spp].origin_tmin.x << " "
        << rg.ray_helper_vec[i * rg.spp].origin_tmin.y << " "
        << rg.ray_helper_vec[i * rg.spp].origin_tmin.z << " "
        << color << " " << color << " " << color
        << std::endl;
    }
  }
  outfile.close();
  printf("done visualization output. \n");

  // Print out the first 10 results to validate by eye
  // for (int rayIndex = 0; rayIndex < 10; rayIndex++)
  // printf("%.2f %d\t", hostHits[rayIndex].t_triId_u_v.x, *(int*)&hostHits[rayIndex].t_triId_u_v.y);
}

void RayTraceManager::debugging(RayGenerator& rg)
{
  std::vector<Hit> hostHits(numRays);
  cudaCheck(cudaMemcpy(hostHits.data(), cudaHits, sizeof(Hit) * numRays, cudaMemcpyDeviceToHost));
  assert(numRays % rg.spp == 0);

  printf("numrays traced: %i \n", (int)numRays);

  uint16_t wrong_ray_output_count = 0;


  for( int i = 0; i < numRays; i++)
  {
    //if (hostHits[i].t_triId_u_v.x == 0.0f && *(int*)&hostHits[i].t_triId_u_v.y == 0)
    {
      wrong_ray_output_count++;

      printf("%d %.2f %d\t", i, hostHits[i].t_triId_u_v.x, *(int*)&hostHits[i].t_triId_u_v.y);
      printf("%f %f %f %f %f %f \n",
      rg.ray_helper_vec[i * rg.spp].origin_tmin.x, rg.ray_helper_vec[i * rg.spp].origin_tmin.y, rg.ray_helper_vec[i * rg.spp].origin_tmin.z,
      rg.ray_helper_vec[i * rg.spp].dir_tmax.x, rg.ray_helper_vec[i * rg.spp].dir_tmax.y, rg.ray_helper_vec[i * rg.spp].dir_tmax.z );
    }

    /*
    */
  }

  printf("wrong_ray_output_count: %d\n", wrong_ray_output_count);
}
