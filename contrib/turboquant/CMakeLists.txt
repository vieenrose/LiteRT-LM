cmake_minimum_required(VERSION 3.16)
project(tq3_engine2 CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Boox Tab Mini C is ARMv8.0-A (no dotprod, no fp16 arith): use
# -DTQ3_MARCH=armv8-a for Android builds.
set(TQ3_MARCH "native" CACHE STRING "-march value")
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=${TQ3_MARCH}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=${TQ3_MARCH}")
find_package(OpenMP REQUIRED)

# LiteRT C API: vendored headers (VibeASR.cpp's trimmed set builds standalone)
# + the stock prebuilt libLiteRt.so from the litert fork.
set(LITERT_INCLUDE "$ENV{HOME}/vibe/VibeASR.cpp/third_party"
    CACHE PATH "dir containing litert/c headers")
set(LITERT_LIB "$ENV{HOME}/litert-fork/litert/prebuilt/linux_x86_64/libLiteRt.so"
    CACHE FILEPATH "prebuilt libLiteRt.so")
set(TQ3_DIR "$ENV{HOME}/turboquant/engine" CACHE PATH "phase 2b tq3.c location")

add_executable(engine2 engine2.cc tq3_attn.cc ${TQ3_DIR}/tq3.c)
target_include_directories(engine2 PRIVATE ${LITERT_INCLUDE} ${TQ3_DIR}
                           ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(engine2 PRIVATE ${LITERT_LIB} m OpenMP::OpenMP_CXX)
get_filename_component(LITERT_LIB_DIR ${LITERT_LIB} DIRECTORY)
set_target_properties(engine2 PROPERTIES BUILD_RPATH ${LITERT_LIB_DIR})
if(ANDROID)
  # NDK: statically link libomp so the binary runs from /data/local/tmp
  target_link_options(engine2 PRIVATE -static-openmp)
endif()
