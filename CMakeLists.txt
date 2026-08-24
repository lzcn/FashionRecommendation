cmake_minimum_required(VERSION 3.21)

project(StarSampling
  VERSION 1.0.0
  DESCRIPTION "Pure C++17 implementation and reproduction of Star Sampling (IJCAI 2017)"
  LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

option(STAR_SAMPLING_NATIVE "Optimize for the host CPU (-march=native)" OFF)
option(STAR_SAMPLING_LTO    "Enable link-time optimization" OFF)
option(STAR_SAMPLING_WERROR "Treat compiler warnings as errors" OFF)

include(CheckIPOSupported)
check_ipo_supported(RESULT star_sampling_ipo_supported OUTPUT star_sampling_ipo_output)

function(star_sampling_configure_target target)
  target_compile_features(${target} PRIVATE cxx_std_17)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
    if(STAR_SAMPLING_WERROR)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    if(STAR_SAMPLING_WERROR)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
    if(STAR_SAMPLING_NATIVE)
      target_compile_options(${target} PRIVATE -march=native)
    endif()
  endif()
  if(STAR_SAMPLING_LTO AND star_sampling_ipo_supported)
    set_target_properties(${target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()
endfunction()

set(STAR_SAMPLING_TOOLS star_sampling prepare_data factorize matrix_benchmark)

foreach(tool IN LISTS STAR_SAMPLING_TOOLS)
  add_executable(${tool} src/${tool}.cpp)
  star_sampling_configure_target(${tool})
endforeach()

enable_testing()
foreach(tool IN ITEMS star_sampling factorize matrix_benchmark)
  add_test(NAME ${tool}_self_test COMMAND ${tool} --self-test)
endforeach()

install(TARGETS ${STAR_SAMPLING_TOOLS} RUNTIME DESTINATION bin)
