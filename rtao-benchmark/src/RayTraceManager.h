#include "helper_math.h"
#include <string>
#include <vector>

class RayGenerator;
class BVHManager;

enum class TraceType {
  Aila,
  CWBVH,
#ifdef OPTIX_PRIME
  OptiXPrime,
#endif
};

class RayTraceManager {
  public:
    RayTraceManager(RayGenerator& rg);
    ~RayTraceManager() {};

    void traceOptiXPrime(RayGenerator& rg);
    void traceAila(RayGenerator& rg);
    void traceCWBVH(RayGenerator& rg);
    void traceCWBVHSingleRay(RayGenerator& rg);
    void traceOptiX(RayGenerator& rg);

    void pathtrace(RayGenerator& rg, TraceType type);
    void ao(RayGenerator& rg, TraceType type);
    void outputToImage(RayGenerator& rg, const std::string& out_image_path);
    void updatePixels(RayGenerator& rg);

    void evaluateAndPrintForPLYVisualization(RayGenerator& rg, const std::string& out_ply_path);
    void debugging(RayGenerator& rg);
    
    uint getNumRays() { return numRays; }
    void setNumRays(uint n);

    void setAnyHit(bool anyhit) { this->anyhit = anyhit; }
    bool getAnyHit(bool anyhit) { return anyhit; };

    void setMaxDepth(int depth) { maxDepth = depth; }
    int getMaxDepth() const { return maxDepth; }

    void setShading(const std::string& shading) { this->shading = shading; }

    void setSortRays(bool sort) { this->sort = sort; }
    bool getSortRays(bool sort) { return sort; };

  private:
    void allocCudaHits(size_t num);

    uint numRays;
    Hit* cudaHits;
    bool anyhit;
    int maxDepth;
    std::string shading;
    bool sort;
    std::vector<float3> pixels;
};