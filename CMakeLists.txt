cmake_minimum_required(VERSION 3.12)
project(estimate-pi LANGUAGES CXX)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

add_executable(estimate-pi main.cpp)
target_link_libraries(estimate-pi OpenMP::OpenMP_CXX MPI::MPI_CXX)