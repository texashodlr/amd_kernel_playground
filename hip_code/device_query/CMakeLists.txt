cmake_minimum_required(VERSION 3.21)
project(device_query LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer hipcc as the compiler:
# cmake -S . -B build -DCMAKE_CXX_COMPILER=hipcc
# (or set CXX=hipcc)
add_executable(device_query src/main.cpp)

# Useful warnings; keep it simple
target_compile_options(device_query PRIVATE -Wall -Wextra -Wpedantic)