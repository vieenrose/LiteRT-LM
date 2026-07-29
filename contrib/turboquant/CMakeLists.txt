cmake_minimum_required(VERSION 3.16)
project(turboquant_tq3 CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# x86 dev default; for Android/arm use -march=armv8-a (nothing needs dotprod).
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
find_package(OpenMP REQUIRED)

# LiteRT C API: point LITERT_INCLUDE at a dir containing the litert/c headers
# (a trimmed standalone-buildable set works) and LITERT_LIB at the stock
# prebuilt libLiteRt.so. No runtime fork/rebuild is needed: the fused custom
# op is registered at runtime via LiteRtAddCustomOpKernelOption.
set(LITERT_INCLUDE "$ENV{HOME}/vibe/VibeASR.cpp/third_party"
    CACHE PATH "dir containing litert/c headers")
set(LITERT_LIB "$ENV{HOME}/litert-fork/litert/prebuilt/linux_x86_64/libLiteRt.so"
    CACHE FILEPATH "prebuilt libLiteRt.so")

get_filename_component(LITERT_LIB_DIR ${LITERT_LIB} DIRECTORY)

# Phase 3 driver: staging/fused auto-detect + voxsum.tq3_attention custom op.
add_executable(engine2 cpp/engine2.cc cpp/tq3_attn.cc cpp/tq3.c)
target_include_directories(engine2 PRIVATE ${LITERT_INCLUDE}
                           ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_link_libraries(engine2 PRIVATE ${LITERT_LIB} m OpenMP::OpenMP_CXX)
set_target_properties(engine2 PROPERTIES BUILD_RPATH ${LITERT_LIB_DIR})

# Phase 2b staging-only engine (A/B reference).
add_executable(engine cpp/engine.cc cpp/tq3.c)
target_include_directories(engine PRIVATE ${LITERT_INCLUDE}
                           ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_link_libraries(engine PRIVATE ${LITERT_LIB} m)
set_target_properties(engine PROPERTIES BUILD_RPATH ${LITERT_LIB_DIR})
