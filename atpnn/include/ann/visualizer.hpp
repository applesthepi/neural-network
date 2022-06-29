#pragma once

#include "components.hpp"

#include <string>
#include <glm/glm.hpp>

namespace ann
{
void visualize_nn(ann::nn_structure* structure, const std::string& file_name);

void draw_line(uint8_t* image, glm::vec<2, int32_t> p0, glm::vec<2, int32_t> p1, float thickness, glm::vec<4, uint8_t> color);
}