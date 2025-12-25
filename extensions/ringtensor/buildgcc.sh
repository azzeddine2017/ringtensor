#!/bin/bash
set -e

# 1️⃣ تحديد المسارات
ROOT="$PWD/../.."
SRC="$PWD"
LIB_DIR="$ROOT/lib"
INCLUDE_DIR="$ROOT/language/include"

mkdir -p "$LIB_DIR"

echo "🏗 Building RingTensor..."

# 2️⃣ ترجمة ring_tensor.c
gcc -c -fpic -O2 "$SRC/ring_tensor.c" -I "$INCLUDE_DIR"

# 3️⃣ إنشاء مكتبة مشتركة
gcc -shared -o "$LIB_DIR/libring_tensor.so" "$SRC/ring_tensor.o"

echo "✅ RingTensor built successfully!"
