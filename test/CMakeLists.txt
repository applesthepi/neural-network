cmake_minimum_required(VERSION 3.15)
project(ATP_NN_TEST)

set(ATP_NN_TEST_SRC "main.cpp")

set(PROJECT_BINARY_DIR ${CMAKE_BINARY_DIR})
add_executable(ATP_NN_TEST ${ATP_NN_TEST_SRC})

if(MSVC)
	source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src FILES ${ATP_NN_TEST_SRC})
	target_compile_options(ATP_NN_TEST PRIVATE "/MP")
	set_target_properties(ATP_NN_TEST PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY
											   $<TARGET_FILE_DIR:ATP_NN_TEST>)
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ATP_NN_TEST)
endif()

add_dependencies(ATP_NN_TEST ATP_NN)

target_link_libraries(
	ATP_NN_TEST PUBLIC
	ATP_NN
)

target_include_directories(
	ATP_NN_TEST PUBLIC
	"${CMAKE_SOURCE_DIR}/atpnn/include"
)