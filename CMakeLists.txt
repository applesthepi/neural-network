cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(USE_FOLDERS ON)
set(CMAKE_BUILD_PARALLEL_LEVEL 12)
enable_language(C CXX ASM)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(ATP_NN_BASE)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(LINUX True)
	message(STATUS "Compiling For Linux")

	find_package(X11 REQUIRED)
	find_package(TBB REQUIRED)

	list(APPEND EXTRA_LIBRARIES ${X11_LIBRARIES})
	list(APPEND EXTRA_INCLUDES ${X11_INCLUDE_DIRS})

    add_compile_options(-msse2)
else()
	set(LINUX False)
endif()

add_subdirectory("atpnn")
add_subdirectory("test")