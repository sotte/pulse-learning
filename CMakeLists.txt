cmake_minimum_required(VERSION 2.8)

## some standard compiler flags
add_definitions(
    -std=c++0x
    -Wall
    -Wnon-virtual-dtor
    -fopenmp
)

## build types
if(CMAKE_BUILD_TYPE MATCHES release|Release|RELEASE)
    add_definitions(-DRELEASE -DNDEBUG -O3)
elseif(CMAKE_BUILD_TYPE MATCHES debug|Debug|DEBUG)
    add_definitions(-DDEBUG -O0)
else()
    message(FATAL_ERROR "Unkown or unspecified Build Type. Use '-DCMAKE_BUILD_TYPE=<type>' with <type> being RELEASE, DEBUG when calling cmake")
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

## ATEM library
add_library(ATEM
    TemporallyExtendedModel.h
    TemporallyExtendedModel.cpp
    lbfgs_codes.h
    lbfgs_codes.cpp
)


## simplepulse library
add_library(simplepulse
    SimpleInterface.cpp
    TemporallyExtendedModel.h
    TemporallyExtendedModel.cpp
    lbfgs_codes.h
    lbfgs_codes.cpp
)

#target_include_directories(ATEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#target_link_libraries(ATEM PUBLIC Qt5::Core)

## Unittests
# add_executable(GTest
#     main.cpp
#     TemporallyExtendedModel.h
#     unit_tests.cpp
# )
# target_link_libraries(GTest PUBLIC
#     -lgtest
#     -larmadillo
#     -llbfgs
#     -lgomp
#     ATEM
# )
