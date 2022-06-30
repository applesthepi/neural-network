cmake_minimum_required(VERSION 3.15)
project(ATP_NN)

file(GLOB_RECURSE ATP_NN_SRC "include/*/*.cpp" "include/*.cpp")

add_library(ATP_NN STATIC ${ATP_NN_SRC})

if(MSVC)
	source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/include/lungo FILES ${ATP_NN_SRC})
	target_compile_options(ATP_NN PRIVATE "/MP")
endif()

# set(SFML_STATIC_LIBRARIES TRUE)
find_package(SFML COMPONENTS network audio graphics window system REQUIRED)

target_include_directories(
	ATP_NN PUBLIC
	"${CMAKE_SOURCE_DIR}/dependencies/stb/include"
	"${CMAKE_SOURCE_DIR}/dependencies/glm"
	# ${SFML_INCLUDE_DIR}
)

target_link_libraries(
	ATP_NN PUBLIC
	sfml-network
	sfml-audio
	sfml-graphics
	sfml-window
	sfml-system
	# glm
	# ${SFML_LIBRARIES}
	# ${SFML_DEPENDENCIES}
)

# message(STATUS "DEBUG PRINT: ${SFML_LIBRARIES}")
# message(STATUS "DEBUG PRINT: ${SFML_DEPENDENCIES}")