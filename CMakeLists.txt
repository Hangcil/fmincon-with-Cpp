cmake_minimum_required(VERSION 3.22)
project(sci_arma)

set(CMAKE_CXX_STANDARD 14)
cmake_minimum_required(VERSION 3.22)
project(arm_test)

set(CMAKE_CXX_STANDARD 14)
set(EXECUTABLE_OUTPUT_PATH ./build)
set(LIBRARY_OUTPUT_PATH ./build)
include_directories(C:/lib/Cxx/armadillo-10.8.2/include/ ./include)
link_directories(C:/lib/Cxx/armadillo-10.8.2/examples/lib_win64/)
add_library(sci ./src/sci_arma.cpp)
add_executable(sci_arma test.cpp)
target_link_libraries(sci_arma libopenblas.lib sci)


