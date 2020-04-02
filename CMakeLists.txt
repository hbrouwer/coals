cmake_minimum_required(VERSION 2.8)

project(coals)

include(CheckIncludeFile)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lm -std=c99 -Wall -O3 -march=native")

## Change these to the correct locations on your machine.
link_directories("/home/brouwerharm/git/coals/lib/svdlibc")
link_libraries("/home/brouwerharm/git/coals/lib/svdlibc/libsvd.a")

add_executable(coals
        src/coals.c)
