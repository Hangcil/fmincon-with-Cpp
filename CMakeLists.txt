
cmake_minimum_required(VERSION 3.22)
project(sci_arma)

set(CMAKE_CXX_STANDARD 14)

set(EXECUTABLE_OUTPUT_PATH ./build)
set(LIBRARY_OUTPUT_PATH ./build)

#the following path should be changed, as it refers to your installed armadillo lib.
include_directories(C:/lib/Cxx/armadillo-10.8.2/include/ ./include)
link_directories(C:/lib/Cxx/armadillo-10.8.2/examples/lib_win64/)


add_library(sci src/Rosenbrock.cpp src/Powell.cpp src/fmincon.cpp src/Powell.cpp src/Powell_modified.cpp src/BFGS.cpp)
add_executable(sci_arma example_without_gra.cpp)

#link libraries
#on windows, to avoid link errors, try place libopenblas.dll and your output '.exe' in the same folder
target_link_libraries(sci_arma libopenblas.lib sci)

