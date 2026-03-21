cmake_minimum_required(VERSION 3.16)
project(cabana_neighbor_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Kokkos REQUIRED)
find_package(Cabana REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

set(PROJECT_LIBS
    Kokkos::kokkos
    Cabana::Core
    Cabana::Grid
)

set(EXAMPLES
    neighfor
    mpi_neighbor
)

foreach(example ${EXAMPLES})
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} PRIVATE ${PROJECT_LIBS})
    
    if(example STREQUAL "mpi_neighbor")
    target_link_libraries(${example} PRIVATE MPI::MPI_CXX)
    target_compile_definitions(${example} PRIVATE USE_MPI=1)
  endif()
endforeach()


