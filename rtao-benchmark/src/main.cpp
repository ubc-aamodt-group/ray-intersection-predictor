#include "RayGenerator.h"
#include "BVHManager.h"
#include "RayTraceManager.h"
#include "Camera.h"
#include "SceneParser.hpp"

#include <iostream>
#include <string>
#include <assert.h>
#include <getopt.h>
#include <cfloat>

int main(int argc, char* argv[])
{
  const std::string base_path = "/home/lucy/sim/ray-intersection-predictor/rtao-benchmark/";
  
  const std::string model_base_path = base_path + "models/";
  const std::string ply_base_path = base_path + "ply_files/";
  const std::string ray_base_path = base_path + "ray_files/";
  const std::string image_base_path = base_path + "images/";
  std::string scene_config_path = base_path + "scene.toml";
  const std::string model_file_type = ".obj";

  std::string model_name = "teapot";
  int debug_ray = 5;
  int num_rays = 0;
  int cwbvh = 0;
  int spp = 1;
  int anyhit = 0;
  int pathtrace = 0;
  int max_depth = 0;
  int sort = 0;
  int num_warmup_rays = 0;
  std::string shading;
  std::string output_name;
  std::string ray_load_path = "";

  static struct option long_options[] =
  {
      {"cwbvh", no_argument, &cwbvh, 1},
      {"model", required_argument, NULL, 'm'},
      {"ray", required_argument, NULL, 'r'},
      {"numRays", required_argument, NULL, 'n'},
      {"spp", required_argument, NULL, 's'},
      {"anyhit", no_argument, &anyhit, 1},
      {"pathtrace", no_argument, &pathtrace, 1},
      {"config", required_argument, NULL, 'c'},
      {"sort", no_argument, &sort, 1},
      {"file", required_argument, NULL, 'f'},
      {"numWarmupRays", required_argument, NULL, 'w'}
  };

  int c;
  while ((c = getopt_long(argc, argv, "m:r:n:s:c:f:w:", long_options, NULL)) != -1)
  {
      switch (c)
      {
          case 'm':
              model_name = optarg;
              break;
          case 'r':
              debug_ray = std::stoi(optarg);
              break;
          case 'n':
              num_rays = std::stoi(optarg);
              break;
          case 's':
              spp = std::stoi(optarg);
              break;
          case 'c':
              scene_config_path = optarg;
              break;
          case 'f':
              ray_load_path = optarg;
              break;
          case 'w':
              num_warmup_rays = std::stoi(optarg);
              break;
      }
  }

  /* Set up Camera */
  Camera camera {};

  if (pathtrace) {
    SceneParser parser(scene_config_path);

    SceneParser::Options parsed_options = parser.get_options();
    cwbvh = cwbvh || parsed_options.cwbvh;
    max_depth = parsed_options.max_depth;
    spp = parsed_options.spp;
    shading = parsed_options.shading;
    sort = sort || parsed_options.sort;

    SceneParser::Model parsed_model = parser.get_model();
    model_name = parsed_model.name;
    output_name = parsed_model.output;

    SceneParser::Camera parsed_camera = parser.get_camera();
    camera = Camera(
      { parsed_camera.position[0], parsed_camera.position[1], parsed_camera.position[2] },
      { parsed_camera.target[0], parsed_camera.target[1], parsed_camera.target[2] },
      { parsed_camera.up[0], parsed_camera.up[1], parsed_camera.up[2] },
      parsed_camera.resolution[0], parsed_camera.resolution[1],
      parsed_camera.fovy
    );
  }


  const std::string out_ply_path = ply_base_path + model_name + ".ply";
  const std::string out_image_path = image_base_path + output_name;
  const std::string model_path = model_base_path + model_name + "/" + model_name + model_file_type;

  /* Set Up the Ray Generator */
  // RayGenerator rg = RayGenerator(4, 4, 0.1, 10); /*spp, spt, t_min, t_max*/
  RayGenerator rg = RayGenerator(camera, spp, 1, 0.1, FLT_MAX); /*spp, spt, t_min, t_max*/

  /* Load the model */
  int triangle_count = rg.loadModelOBJ(model_path);
  assert(triangle_count > 0);
  printf("Done loading obj. Tri count: %i \n", triangle_count);

  /*Build the acceleration structures*/
  BVHManager bvh_manager = BVHManager();
  
  #ifdef MAGIC
  if (cwbvh) bvh_manager.newBuildCWBVH(rg);
  #else
  if (cwbvh) bvh_manager.buildCWBVH(rg);
  #endif
  else bvh_manager.buildBVH2(rg);

  if (!pathtrace) {
    if (ray_load_path == "") {
      std::cout << "No ray file specified, generating..." << std::endl;
      /* Generate rays - uses points and normals -> this takes a while */
      int spp_count;
      spp_count = rg.generatePointsAndNormals_spt(1000000);
      spp_count = rg.generate_detangled_spp();
      printf("Done generating rays detangled. SPP count: %i \n", spp_count);
      // rg.saveRaysToFile(ray_base_path, model_name + "_detangled_random");
    } else {
      std::cout << "Reading rays from: " << ray_load_path << std::endl;
      rg.readRaysFromFile(ray_load_path, UINT32_MAX);
    }

    /* Apply ray sorting + save rays */
    if (sort) {
      rg.raySorting(rg.hash);
    }
    rg.split_warmup_measurement_rays(num_warmup_rays);

    /* Upload rays to the GPU*/
    #ifdef DEBUG
    printf("Uploading single ray (%d)\n", debug_ray);
    rg.uploadSingleRayToGPU(debug_ray);
    #else
    rg.uploadRaysToGPU();
    #endif
  }

  /* Trace the rays */
  RayTraceManager rt_manager = RayTraceManager(rg);

  if (pathtrace) {
    rt_manager.setMaxDepth(max_depth);
    rt_manager.setShading(shading);
    rt_manager.setSortRays(sort);
    if (shading == "ao") {
      rt_manager.ao(rg, cwbvh ? TraceType::CWBVH : TraceType::Aila);
    } else {
      rt_manager.pathtrace(rg, cwbvh ? TraceType::CWBVH : TraceType::Aila);
    }
    rt_manager.outputToImage(rg, out_image_path);
  } else {
    if (num_rays != 0) rt_manager.setNumRays(num_rays);

    printf("Tracing %s hit rays.\n", anyhit ? "any" : "closest");
    rt_manager.setAnyHit(anyhit);

    if (cwbvh) rt_manager.traceCWBVH(rg);
    else rt_manager.traceAila(rg);

    #ifdef OPTIX_PRIME
    rt_manager.traceOptiXPrime(rg);
    #endif

    if (rg.has_measurement_rays()) {
      rg.load_measurement_rays();
      rt_manager.setNumRays(rg.getRayCount());
      rg.uploadRaysToGPU();

      if (cwbvh) rt_manager.traceCWBVH(rg);
      else rt_manager.traceAila(rg);

      #ifdef OPTIX_PRIME
      rt_manager.traceOptiXPrime(rg);
      #endif
    }

    /* Print info for visualizing the rays */
    rt_manager.evaluateAndPrintForPLYVisualization(rg, out_ply_path);

    // rt_manager.debugging(rg); // for debugging - prints output of kernels into command line

    printf("Done, traced %i rays \n", rt_manager.getNumRays());
  }

  return 0;
}

