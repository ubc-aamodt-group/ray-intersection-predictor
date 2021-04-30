#include "Camera.h"

Camera::Camera(const float3& position,
               const float3& target,
               const float3& up,
               uint width,
               uint height,
               float fovy)
    : position(position), target(target), up(up), width(width), height(height), fovy(fovy) {
    float3 w = normalize(position - target);
    float3 u = normalize(cross(up, w));
    float3 v = cross(w, u);

    basis = matrix4x4::From3Cols(make_float4(u, 0.0f), make_float4(v, 0.0f), make_float4(w, 0.0f));

    const auto to_rad = [](float deg) { return deg / 180.0f * M_PI; };

    half_fovy = to_rad(fovy) / 2.0f;
    half_fovx = half_fovy * width / height;
    half_width = width / 2.0f;
    half_height = height / 2.0f;
}

Ray Camera::create_primary_ray(float x, float y, float t_min, float t_max) const {
    float alpha = tan(half_fovx) * (x - half_width) / half_width;
    float beta = tan(half_fovy) * (half_height - y) / half_height;

    float3 direction = normalize(basis.transformVector3(make_float3(alpha, beta, -1.0f)));

    Ray ray;
    ray.make_ray(
        position.x, position.y, position.z,
        direction.x, direction.y, direction.z,
        t_min, t_max);

    return ray;
}