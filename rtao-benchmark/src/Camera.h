#pragma once

#include "helper_math.h"

class Camera {
public:
    Camera() = default;

    Camera(const float3& position,
           const float3& target,
           const float3& up,
           uint width,
           uint height,
           float fovy);

    Ray create_primary_ray(float x, float y, float t_min, float t_max) const;

    uint get_width() const { return width; }
    uint get_height() const { return height; }
    float3 get_position() const { return position; }

private:
    float3 position;
    float3 target;
    float3 up;
    uint width;
    uint height;
    float fovy;

    matrix4x4 basis;
    float half_fovy;
    float half_fovx;
    float half_width;
    float half_height;
};