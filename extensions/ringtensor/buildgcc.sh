#!/bin/bash
set -e

# 1️⃣ تحديد المسارات
ROOT="$PWD/../.."
SRC="$PWD"
LIB_DIR="$ROOT/lib"
INCLUDE_DIR="$ROOT/language/include"

mkdir -p "$LIB_DIR"

echo "🏗 Building RingTensor..."

gcc -shared -o libring_tensor.so -O3 -fPIC -fopenmp -DUSE_OPENCL ring_tensor.c -I ../../language/include -L ../../lib -lring -lOpenCL

echo "✅ RingTensor built successfully!"
