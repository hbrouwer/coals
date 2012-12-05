cmake_minimum_required(VERSION 2.8)

project(coals)

include(CheckIncludeFile)

set(CMAKE_C_FLAGS) "${CMAKE_C_FLAGS} -std=c99 -pendantic -Wall -O3 -march=core2")

link_directories("/Users/harm/git/coals/lib/svdlibc")
link_libraries("/Users/harm/git/coals/lib/svdlibc/libsvd.a")

add_executable(coals
        src/coals.c)
