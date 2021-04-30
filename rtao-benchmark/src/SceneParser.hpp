#pragma once

#include <toml11/toml.hpp>

class SceneParser {
public:
    struct Options {
        bool cwbvh;
        int max_depth;
        int spp;
        std::string shading;
        bool sort;
    };

    struct Model {
        std::string name;
        std::string output;
    };

    struct Camera {
        std::array<float, 3> position;
        std::array<float, 3> target;
        std::array<float, 3> up;
        std::array<unsigned int, 2> resolution;
        float fovy;
    };

    SceneParser(const std::string& path);

    Options get_options() const;
    Model get_model() const;
    Camera get_camera() const;

private:
    toml::value parsed_data;
};
