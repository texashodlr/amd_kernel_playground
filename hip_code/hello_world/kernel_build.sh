#!/bin/bash
mkdir -p build
cmake -S . -B build -DCMAKE_CXX_COMPILER=hipcc
cmake --build build -j