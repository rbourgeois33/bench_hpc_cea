cmake_minimum_required(VERSION 3.25.2)
project(matvec_bw LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Backend selection. Default = CUDA for an RTX 6000 Ada (Ada Lovelace, CC 8.9).
# OPENMP / SERIAL are only there for a quick CPU smoke test.
# ---------------------------------------------------------------------------
set(MATVEC_BACKEND "CUDA" CACHE STRING "Kokkos device backend: CUDA | OPENMP | SERIAL")
set_property(CACHE MATVEC_BACKEND PROPERTY STRINGS CUDA OPENMP SERIAL)

option(MATVEC_KERNEL_REPORT
  "CUDA only: pass '-Xptxas -v' to nvcc to print stack frame / spill / register usage per kernel" OFF)

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
if(MATVEC_BACKEND STREQUAL "CUDA")
  set(Kokkos_ENABLE_CUDA  ON CACHE BOOL "" FORCE)
  set(Kokkos_ARCH_ADA89   ON CACHE BOOL "" FORCE)   # RTX 6000 Ada
elseif(MATVEC_BACKEND STREQUAL "OPENMP")
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
elseif(NOT MATVEC_BACKEND STREQUAL "SERIAL")
  message(FATAL_ERROR "MATVEC_BACKEND must be CUDA, OPENMP or SERIAL (got '${MATVEC_BACKEND}')")
endif()

# ---------------------------------------------------------------------------
# Kokkos 5.2.2 (latest release as of Sept. 2026), fetched as a release tarball.
# To use a local checkout instead: -DFETCHCONTENT_SOURCE_DIR_KOKKOS=/path/to/kokkos
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
  kokkos
  URL      https://github.com/kokkos/kokkos/releases/download/5.2.2/kokkos-5.2.2.tar.gz
  URL_HASH SHA256=d6557aaef39302282a15f9c770433d1fcdf4e961dfd6d9ed726b9d0d0f546b9f
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(kokkos)
# With CUDA and a plain host compiler (g++), Kokkos installs kokkos_launch_compiler
# as a global compile rule, so our sources go through nvcc_wrapper automatically.

add_executable(matvec_bench main.cpp)
target_compile_features(matvec_bench PRIVATE cxx_std_20)
target_link_libraries(matvec_bench PRIVATE Kokkos::kokkos)

if(MATVEC_KERNEL_REPORT AND MATVEC_BACKEND STREQUAL "CUDA")
  target_compile_options(matvec_bench PRIVATE "SHELL:-Xptxas -v")
endif()