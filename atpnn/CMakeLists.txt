cmake_minimum_required(VERSION 3.15)
project(ATP_NN)

file(GLOB_RECURSE ATP_NN_SRC "include/*/*.cpp" "include/*.cpp")

add_library(ATP_NN STATIC ${ATP_NN_SRC})

if(MSVC)
	source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/include/lungo FILES ${ATP_NN_SRC})
	target_compile_options(ATP_NN PRIVATE "/MP")
endif()