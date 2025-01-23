cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch-jit-example)

set(CMAKE_PREFIX_PATH "/home/gmihmih29/wrk/tools/libtorch")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

find_package(Torch REQUIRED)

file(GLOB BASE_SRC CONFIGURE_DEPENDS
    "src/*.hpp"
    "src/*.cpp"
)
add_executable(${PROJECT_NAME} ${BASE_SRC})
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${TORCH_LIBRARIES}
)
