cmake_minimum_required(VERSION 3.0)

## some standard compiler flags
add_definitions(
    -std=c++0x
    -Wall
    -Wnon-virtual-dtor
)

## build types
if(CMAKE_BUILD_TYPE MATCHES release|Release|RELEASE)
    add_definitions(-DRELEASE -DNDEBUG -O3)
elseif(CMAKE_BUILD_TYPE MATCHES debug|Debug|DEBUG)
    add_definitions(-DDEBUG -O0)
else()
    message(FATAL_ERROR "Unkown or unspecified Build Type. Use '-DCMAKE_BUILD_TYPE=<type>' with <type> being RELEASE, DEBUG when calling cmake")
endif()

## ATEM library
add_library(ATEM
	AbstractTemporallyExtendedModel.h
	TemporallyExtendedModel.h
	TemporallyExtendedModel.cpp
)
#target_include_directories(ATEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#target_link_libraries(ATEM PUBLIC Qt5::Core)

## Unittests
add_executable(GTest
	main.cpp
	AbstractTemporallyExtendedModel.h
	TemporallyExtendedModel.h
	unit_tests.cpp
)
target_link_libraries(GTest PUBLIC
    -lgtest
    ATEM
)
