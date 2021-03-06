#pragma once

#include "components.hpp"

#include <string>
#include <glm/glm.hpp>

namespace ann
{
void visualize_nn(ann::nn_structure* structure, const std::string& file_name);
}