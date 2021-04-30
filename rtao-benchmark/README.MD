# Ray-Traced Ambient Occlusion Benchmark for Hybrid Rendering Workloads

## Notes:
- Update `base_path` in `main.cpp`
- Add `.obj` models to `models` folder, e.g. for model `teapot`, should be in `models/teapot/teapot.obj`
- Create a `ply_files` folder to see results
- Create an `images` folder to see rendered images
- Run `./CWBVH` for original code; run `./magic_CWBVH` to use the ray trace function in gpgpu-sim

### For GI:
- Run `./CWBVH --pathtrace [--cwbvh]`
- This uses config specified in `.toml` config files. Most arguments from the CLI will be ignored
  and overridden with settings from these files. The exception is `--cwbvh`, which the CLI overrides.
- You can optionally specify the path to a config file using `./CWBVH --pathtrace --config <toml>`.
  If not specified, it will use the `scene.toml` in the root directory.

## For AO:
- Run `./magic_CWBVH --anyhit`
- There are a few flags available in `main.cpp`
- e.g. `./magic_CWBVH -m sponza --anyhit -f ../ray_files/sponza_sorted_40000.ray_file -w 10000` will run the sponza scene, loading 40000 rays from the ray file, use 10000 rays for the warmup kernel, and 30000 rays for the measurement kernel


# Original README:

![Image: RTAO Dragon](https://github.com/bftf/CWBVH/tree/master/images/dragon_readme.png)

![Image: RTAO Hairball](https://github.com/bftf/CWBVH/tree/master/images/hairball_readme.png)

# Build

Before building:
---------
Requires CUDA SDK v10.0 (the CUDA SDK version can be easily downgraded), tbb and embree3.
Requires OptiX Prime v5.1

The build process requires the use of a Makefile. The advantage of the makefile is that CUDA can be linked statically which is a requirement for running the RTAO benchmark on GPGPU-Sim.  The CMake is currently not working. 

Edit the following lines in your Makefile to use the paths to your libraries:

USING_OPTIX=0# 1 enables OptiX
OPTIX_INCLUDE_DIR = ~/Documents/RayTracing/NVIDIA-OptiX-SDK-5.1.0-linux64/include
OPTIX_LIB = /home/francois/Documents/RayTracing/NVIDIA-OptiX-SDK-5.1.0-linux64/lib64/ # make sure this is in LD_LIBRARY_PATH

EMBREE_INCLUDE_DIR = ~/Documents/RayTracing/embree/include
EMBREE_LIB = ~/Documents/RayTracing/embree/build

In the main.cpp you have to change the path to the models that you are trying to render. A great source for the models is Morgan McGuire's repo: https://casual-effects.com/g3d/data10/index.html Currently only obj input models are supported.

When reading a model you need to edit the base_path and the model_name to match the location of the model you want to load:
const std::string model_base_path = "/home/francois/Documents/RayTracing/models/";
const std::string model_name = "teapot";
const std::string obj_path_addition = "teapot.obj";

The final path for this example will be: /home/francois/Documents/RayTracing/models/teapot


Build:
---------
To build:
cd build_make;
make
./CWBVH

Currently only Linux has been tested.

# Ambient Occlusion

Ray Generation
-------------
Ray are generated on a per triangle basis. 
The user can input a given Samples Per Triangle (SPT) value and a Samples per Pixel (SPP) value when instantiating the RayGenerator class in main.cpp.

Ray origins are generated using a random distribution over the triangle.
Ray directions are generated using cosine-weighted stochastic hemisphere sampling. 

Ray Tracing
-------------
We support 3 ray tracers:
1. Architecture Considerations for Tracing Incoherent Rays (https://research.nvidia.com/sites/default/files/pubs/2010-06_Architecture-Considerations-for/aila2010hpg_paper.pdf)
2. Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs (http://research.nvidia.com/publication/2017-07_Efficient-Incoherent-Ray)
3. OptiX Prime (version 5.2 is tested)

We hope to support hardware accelerated OptiX in the future.

Output Visualization
-------------
We print out a .ply file that represents a pointcloud of the ambient occlusion results. 
We recommend loading the .ply file into Blender (https://www.blender.org/) and then visualizing it with the PointCouds plugin for Blender (https://github.com/uhlik/bpy#point-cloud-visualizer-for-blender-280, https://www.youtube.com/watch?v=eXct_7k779Q)

# Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs
--------
All credits for the implementation of "Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs" (http://research.nvidia.com/publication/2017-07_Efficient-Incoherent-Ray) go to AlanIWBFT, this work is based on his repo at AlanIWBFT/CWBVH.
