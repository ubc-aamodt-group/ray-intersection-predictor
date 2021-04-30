
#include <memory>

class RayGenerator;

class BVHManager {
  public:
    BVHManager() {};
    ~BVHManager() {};

    void buildBVH2(RayGenerator& rg);
    void buildCWBVH(RayGenerator& rg);
    void newBuildCWBVH(RayGenerator& rg);

  private:

    friend class RayTraceManager;
};