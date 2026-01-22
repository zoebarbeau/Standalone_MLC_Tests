Cabana and Kokkos must be built first. See Cabana wiki for instructions.
export CABANA_INSTALL=~/Cabana/build/install
    # TODO: YOU, THE USER, SHOULD CHANGE THESE TO YOUR DESIRED PATHS
    export KOKKOS_SRC_DIR=`pwd`/kokkos
    export KOKKOS_INSTALL_DIR=$KOKKOS_SRC_DIR/build/install
```
    cd ./kokkos
    mkdir build
    cd build
    cmake \
      -D CMAKE_BUILD_TYPE="Release" \
      -D CMAKE_INSTALL_PREFIX=$KOKKOS_INSTALL_DIR \
      -D Kokkos_ENABLE_SERIAL=ON \
      -D Kokkos_ENABLE_OPENMP=ON \
      -D Kokkos_ENABLE_CUDA=ON \
      -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
      -D Kokkos_ENABLE_IMPL_VIEW_LEGACY=ON \
      \
      .. ;
    make install
    cd ../.. # Go back to top level dir  
```

```
    # TODO: YOU, THE USER, SHOULD CHANGE THESE TO YOUR DESIRED PATHS
    export KOKKOS_SRC_DIR=`pwd`/kokkos
    export KOKKOS_INSTALL_DIR=`pwd`/kokkos/build/install
    export CABANA_INSTALL_DIR=`pwd`/Cabana/build/install

    cd ./Cabana
    mkdir build
    cd build
    cmake \
     -D CMAKE_BUILD_TYPE="Debug" \
     -D CMAKE_PREFIX_PATH=$KOKKOS_INSTALL_DIR \
     -D CMAKE_INSTALL_PREFIX=$CABANA_INSTALL_DIR \
     -D Cabana_REQUIRE_CUDA=ON \
     .. ;
    make install
```
 ```   
cd standalone
mkdir build
cd build
cmake ..   -DCMAKE_BUILD_TYPE=Release   -DCMAKE_PREFIX_PATH="$KOKKOS_INSTALL_DIR;$CABANA_INSTALL_DIR"
make 
```
