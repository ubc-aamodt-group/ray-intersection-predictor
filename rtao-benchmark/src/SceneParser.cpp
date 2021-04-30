#include "SceneParser.hpp"

SceneParser::SceneParser(const std::string& path) : parsed_data(toml::parse(path)) {}

SceneParser::Options SceneParser::get_options() const {
    bool cwbvh = toml::find<bool>(parsed_data, "options", "cwbvh");
    int max_depth = toml::find<int>(parsed_data, "options", "max_depth");
    int spp = toml::find<int>(parsed_data, "options", "spp");
    std::string shading = toml::find<std::string>(parsed_data, "options", "shading");
    bool sort = toml::find<bool>(parsed_data, "options", "sort");

    return { cwbvh, max_depth, spp, shading, sort };
}

SceneParser::Model SceneParser::get_model() const {
    std::string name = toml::find<std::string>(parsed_data, "model", "name");
    std::string output = toml::find<std::string>(parsed_data, "model", "output");

    return { name, output };
}

SceneParser::Camera SceneParser::get_camera() const {
    std::array<float, 3> position = toml::find<std::array<float, 3>>(parsed_data, "camera", "position");
    std::array<float, 3> target = toml::find<std::array<float, 3>>(parsed_data, "camera", "target");
    std::array<float, 3> up = toml::find<std::array<float, 3>>(parsed_data, "camera", "up");
    std::array<unsigned int, 2> resolution = toml::find<std::array<unsigned int, 2>>(parsed_data, "camera", "resolution");
    float fovy = toml::find<float>(parsed_data, "camera", "fovy");

    return { position, target, up, resolution, fovy };
}
