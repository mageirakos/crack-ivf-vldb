cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(AVTree LANGUAGES CXX)

add_executable(avtree main_knn.cpp tree/AVtree.cpp)
target_compile_features(avtree PUBLIC cxx_std_17 )
target_compile_definitions(avtree PUBLIC L2)
target_compile_options(avtree PUBLIC -O3 -mavx -march=native -Wall -Wextra -Wpedantic -L$ENV{CONDA_PREFIX}/lib)

set(HIGHFIVE_USE_BOOST Off)
add_subdirectory(../HighFive HighFive)
target_link_libraries(avtree HighFive)
target_link_libraries(avtree hdf5)
target_link_libraries(avtree hdf5_cpp)