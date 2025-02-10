cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch-jit-example)

set(CMAKE_PREFIX_PATH "/home/gmihmih29/wrk/tools/libtorch")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/resources/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

find_package(Torch REQUIRED)
find_package(OpenCV 4.2.0 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_definitions(-DUSE_OPEN_CV)

file(GLOB TIMER_SRC CONFIGURE_DEPENDS
    "src/timer/*.hpp"
    "src/timer/*.cpp"
)
add_library(TIMER ${TIMER_SRC})

file(GLOB BASIC_SRC CONFIGURE_DEPENDS
    "src/basic/*.hpp"
    "src/basic/*.cpp"
)
add_executable("basic" ${BASIC_SRC})
target_include_directories("basic" PUBLIC
    src/
)
target_link_libraries("basic" PUBLIC
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    TIMER
)

file(GLOB MINI_UNET_SRC CONFIGURE_DEPENDS
    "src/mini_unet/*.hpp"
    "src/mini_unet/*.cpp"
)
add_executable("mini_unet" ${MINI_UNET_SRC})
target_include_directories("mini_unet" PUBLIC
    src/
)
target_link_libraries("mini_unet" PUBLIC
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    TIMER
)

file(GLOB FUNCTIONS_SRC CONFIGURE_DEPENDS
    "src/functions/*.hpp"
    "src/functions/*.cpp"
)
add_executable("functions" ${FUNCTIONS_SRC})
target_include_directories("functions" PUBLIC
    src/
)
target_link_libraries("functions" PUBLIC
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    TIMER
)
