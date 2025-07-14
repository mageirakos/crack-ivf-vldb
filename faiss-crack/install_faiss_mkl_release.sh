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
# NOTE: you need intel mkl installed in mamba
# MAMBA_MKL_PATH=$(dirname $(dirname $(which conda)))/lib
# MAMBA_ENV_PATH=$(dirname $(dirname $(which mamba)))/envs/$(basename $(dirname $(dirname $(which mamba))))/lib

# MAMBA_ENV_PATH=$CONDA_PREFIX/lib/
# # Print the MAMBA_ENV_PATH to verify the correct path
# echo "MAMBA_ENV_PATH is: $MAMBA_ENV_PATH"

# # List the contents of the lib directory to check if MKL libraries are present
# echo "Contents of the lib directory:"
# ls $MAMBA_ENV_PATH

# # Add to LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$MAMBA_ENV_PATH:$LD_LIBRARY_PATH
# # Verify LD_LIBRARY_PATH to ensure it's set correctly
# echo "LD_LIBRARY_PATH is: $LD_LIBRARY_PATH"


#  Find where mamba installed MKL
MAMBA_MKL_PATH=$(dirname $(dirname $(which conda)))/lib

# Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MAMBA_MKL_PATH:$LD_LIBRARY_PATH

echo "$LD_LIBRARY_PATH"
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
    # (cd build-release/faiss/python && python setup.py install)
    pip install build-release/faiss/python
    echo "Python bindings installed successfully"
fi

echo "Build completed successfully!"