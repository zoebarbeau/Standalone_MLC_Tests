cmake_minimum_required(VERSION 3.16)
project(cabana_neighbor_test LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Kokkos and Cabana
find_package(Kokkos REQUIRED)
find_package(Cabana REQUIRED)

add_executable(neighbor_test neighfor.cpp)

target_link_libraries(neighbor_test
    Kokkos::kokkos
    Cabana::Core
    Cabana::Grid
)

# Optional: enable warnings
target_compile_options(neighbor_test PRIVATE -Wall -Wextra)

