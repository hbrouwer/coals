cmake_minimum_required(VERSION 2.8)

project(coals)

include(CheckIncludeFile)

set(CMAKE_C_COMPILER "/opt/local/bin/gcc-mp-4.8")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall -O3 -march=core2")

link_directories("/Users/harm/git/coals/lib/svdlibc")
link_libraries("/Users/harm/git/coals/lib/svdlibc/libsvd.a")

add_executable(coals
        src/coals.c)
