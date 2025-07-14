[faiss] , [crack-ivf]

Folder Structure:
- `/benchs` : contains code to recreate paper benchmarks
- `conda` : i think contains the conda builds of faiss for when you `conda install faiss-cpu` 
- `contrib` : varius wrappers and contributions from people, typically in python
	- helper functions to inspect internal classes 
- `demos`: i think a more complex version of the tutorial
- `faiss` : main code 
- `misc` : just has `test_blas.cpp` file to test some kernels
- `perf_tests` : similar to misc
- `tests` : testing code for all/most of faiss
- `tutorial` : self-explanatory

---
If you want to install faiss for python, from source :
- install:
```
(cd build-release/faiss/python && python setup.py install)
```
- if you've not built it:
```
make -C your-local-build -j swigfaiss
```
---
#### **Step 1: Configure Requirements**

I careated `mamba create -n crack-ivf-dev` and you can `mamba activate crack-ivf-dev` for development. To translate the above:
- if needed: `conda update -n base -c defaults certifi`
- `mamba install -y python=3.9 numpy=1.26.4 scipy pytest swig gcc gxx libopenblas llvm-openmp wget` 
Not sure if these are needed:
- `sudo apt update && sudo apt install -y mc git sudo`
- `mamba env export > crack-ivf-dev.yaml`
	- export env for reproducibility
- (optional) `wget -qO- "https://cmake.org/files/v3.26/cmake-3.26.5-linux-x86_64.tar.gz" | sudo tar --strip-components=1 -xz -C /usr/local`
	- download specific cmake version if you need it (already had one)
- `cmake --version`
	- check which version you have

#### Updating cmake when you don't have sudo (eg. hacc-box-3)
activate enviroment
- `mamba activate crack-ivf-dev`
download cmake version 
- `wget -qO- "https://cmake.org/files/v3.26/cmake-3.26.5-linux-x86_64.tar.gz" -O cmake-3.26.5-linux-x86_64.tar.gz`
extract content under ~/cmake where i have permissions...
- `mkdir -p ~/cmake`
- `tar --strip-components=1 -xz -C ~/cmake -f cmake-3.26.5-linux-x86_64.tar.gz`
add cmake to path
- `export PATH=~/cmake/bin:$PATH`
to make it persistent across sessions add it to .bashrc
- `echo 'export PATH=~/cmake/bin:$PATH' >> ~/.bashrc`
- `source ~/.bashrc`
now test which version you have:
- `cmake --version`
#### **Step 2: Configure Debug Build**

**NOTE**: you need to `make -j8` after every code change

1. Create a `debug` build directory:   
``` bash
cmake -B build-debug . \
    -DCMAKE_BUILD_TYPE=Debug \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_TESTING=ON \
    -DFAISS_USE_LTO=OFF
```
**Explanation of Flags:**
- `-DCMAKE_BUILD_TYPE=Debug`: Enables debug symbols for better debugging.
- `-DFAISS_ENABLE_GPU=ON`: Enables GPU indices (adjust to `OFF` if not needed).
- `-DFAISS_ENABLE_PYTHON=ON`: Builds Python bindings.
- `-DBUILD_TESTING=ON`: Ensures tests are built to validate functionality.
- `-DFAISS_USE_LTO=OFF`: Disables Link-Time Optimization, which is unsuitable for debugging.

2. Build the `debug` version:
only builds faiss:
```
make -C build-debug -j faiss
```

#### **Step 3: Configure Release Build**
##### Release w/ python bindings w/ MKL library w/ my custom changes... (==THIS WORKS NICE==)

FAISS had trouble finding MKL or I was just not setting it up properly. You can use find_library() from cmake but you need to pass MKL into LD_LIBRARY_PATH either in .bashrc or since I have a mamba enviroment and not always going to be using mkl ( I also lack sudo ).
I use the following in hacc-box-03.inf.ethz.ch


chmod +x build_faiss_mkl_release.sh
`./build_faiss_mkl_release.sh` <<<< build from scratch
- comment in/out what version you want to install...
==build_faiss_mkl_release.sh== : 
``` bash
#!/bin/bash

# usage:
# Make the script executable:
# $ chmod +x install_faiss_mkl_release.sh
# Build without Python support:
# $ ./install_faiss_mkl_release.sh
# Or build with Python support:
# $ ./install_faiss_mkl_release.sh --python

# Function to print usage
print_usage() {
    echo "Usage: $0 [--python]"
    echo "Build faiss with optional Python support"
    echo "  --python    Build with Python support"
}

# Parse command line arguments
BUILD_PYTHON=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --python)
            BUILD_PYTHON=1
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check if conda/mamba environment is activated
if [[ -z "${CONDA_PREFIX}" ]]; then
    echo "Error: No conda/mamba environment detected. Please activate your environment first."
    exit 1
fi

# Find where mamba installed MKL
MAMBA_MKL_PATH=$(dirname $(dirname $(which conda)))/lib

# Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MAMBA_MKL_PATH:$LD_LIBRARY_PATH

# Remove existing build directory
rm -rf build-release

# Base cmake flags
CMAKE_FLAGS=(
    -DFAISS_ENABLE_GPU=OFF
    -DBUILD_SHARED_LIBS=OFF
    -DCMAKE_BUILD_TYPE=Release
    # -DFAISS_OPT_LEVEL=avx2
)

# Add Python flags if requested
if [[ $BUILD_PYTHON -eq 1 ]]; then
    echo "Building with Python support..."
    CMAKE_FLAGS+=(
        -DFAISS_ENABLE_PYTHON=ON
        -DPython_EXECUTABLE=$CONDA_PREFIX/bin/python
    )
fi

# Run cmake with all flags
cmake -B build-release . "${CMAKE_FLAGS[@]}"

# Build faiss
echo "Building faiss..."
make -C build-release -j faiss

# Build and install Python bindings if requested
if [[ $BUILD_PYTHON -eq 1 ]]; then
    echo "Building Python bindings..."
    make -C build-release -j swigfaiss
    (cd build-release/faiss/python && python setup.py install)
    echo "Python bindings installed successfully"
fi

echo "Build completed successfully!"```


##### (skip this section) MKL/OMP for some reason don't work this way:

I had issues with OMP working through the python binding & also had issues with MKL.
These were fixed by running the above 

(skip) MLK in python does not work for some reason (if install from source)
Have intel MKL install in the mamba enviroment:
- `mamba install -c conda-forge mkl`
To get path to mkl libs:
- `mamba list mkl`
- `cd $CONDA_PREFIX/lib`
- `which python`

You may need to install : `mamba install gflags`
- it was an error thrown by one of the cmake if DBUILD_TESTING=ON

2. Create a `release` build directory:   
``` bash
cmake -B build-release . \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_ROCM=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_TESTING=OFF \
    -DFAISS_USE_LTO=ON \
    -DFAISS_OPT_LEVEL=avx2 \
    -DBLA_VENDOR=Intel10_64_dyn \
    -DMKL_LIBRARIES=$CONDA_PREFIX/lib \
    -DPython_EXECUTABLE=$CONDA_PREFIX/bin/python
```

**Explanation of Flags:**
- `-DCMAKE_BUILD_TYPE=Release`: Optimizes the build for performance.
- `-DFAISS_USE_LTO=ON`: Enables Link-Time Optimization for additional performance.
- `-DFAISS_OPT_LEVEL=avx2`: Optimizes for AVX2 instruction set (adjust to `generic`, `avx512`, etc., based on your system).

2. Build the `release` version:
```
make -C build-release -j faiss
$ make -C build-release -j faiss_avx2
$ make -C build-release -j faiss_avx512
$ make -C build-release -j faiss_avx512_spr
```

mini test:
 ```
make -C build-release -j demo_ivfpq_indexing
./build-release/demos/demo_ivfpq_indexing
```

#### Step 4: Test Builds
- `make -C build-debug test`
- `make -C build-release test`

Optional : Build Python Bindings:

$ make -C build -j swigfaiss
$ (cd build/faiss/python && python setup.py install)

Debug version:
```
make -C build-debug -j swigfaiss
cd build-debug/faiss/python
python setup.py install
```
Release version:
```
make -C build-release -j swigfaiss
(cd build-release/faiss/python && python setup.py install)

cd build-release/faiss/python
python setup.py install
```
##### Flags to build for  ROCm:
`-DFAISS_ENABLE_ROCM=ON` in order to enable building GPU indices for AMD GPUs.
`-DFAISS_ENABLE_GPU` must be `ON` when using this option. (possible values are `ON` and `OFF`),


---

# Building/Running code

By default demos and bench etc. are not part of the default build because they have this EXCLUDE in their CMakeLists.txt file:
```
add_executable(demo_imi_flat EXCLUDE_FROM_ALL demo_imi_flat.cpp)
```
### To add your own script:
I did not add `EXCLUDE_FROM_ALL` since I want it to be part of the default build
``` 
add_executable(demo_my_example demo_my_example.cpp)
target_link_libraries(demo_my_example PRIVATE faiss)
```
from inside the build folder:
```
cmake <w/ all the flags you had>
make -j bench_crack_ivf
```
### Run code:

Let's say it's bench_crack_ivf.cpp under /benchs:
- `make -j bench_crack_ivf`
- `./build-debug/benchs/bench_crack_ivf`

#### Run in vscode debugger:
You need to create a `lunch.json` file under `.vscode` and set it up for every file you wish to debug:
In this example I am running bench_crack_ivf under debug with gdb debugger found in /usr/bin/gdb
``` json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "[DEBUG](crack-ivf-dev) bench_crack_ivf.cpp",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build-debug/benchs/bench_crack_ivf",
            // "args": [
            //     "--argument", "value",
            // ],
            "stopAtEntry": false,
            "cwd": "${fileDirname}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "miDebuggerPath": "/usr/bin/gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                },
                {
                    "description": "Set Disassembly Flavor to Intel",
                    "text": "-gdb-set disassembly-flavor intel",
                    "ignoreFailures": true
                }
            ]
        }

    ]
}
```


---
# Other

**Do not use dockerfile, if you want custom implementation. It clones remote which will not have your custom code...** 
Dockerfile under `/faiss/cppcontrib/docker_dev` which can be useful when running in a new system and as a guide on how to install.


``` Dockerfile
FROM ubuntu:22.04

RUN apt update && apt install -y python3 python3-pip gcc g++ mc git swig sudo libomp-dev libopenblas-dev wget
RUN pip3 install numpy==1.26.4 scipy pytest
RUN cd /root && git clone https://github.com/facebookresearch/faiss
RUN wget -qO- "https://cmake.org/files/v3.26/cmake-3.26.5-linux-x86_64.tar.gz" | sudo tar --strip-components=1 -xz -C /usr/local
RUN cd /root/faiss && /usr/local/bin/cmake -B build -DFAISS_ENABLE_GPU=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release .
RUN cd /root/faiss && make -C build -j 8 faiss
RUN cd /root/faiss && make -C build -j 8 swigfaiss
RUN cd /root/faiss/build/faiss/python && python3 setup.py install
```
