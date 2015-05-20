cmake_minimum_required(VERSION 3.0)

add_definitions(-std=c++0x)

add_library(ATEM
	AbstractTemporallyExtendedModel.h
	TemporallyExtendedModel.h
	TemporallyExtendedModel.cpp
)

#target_include_directories(ATEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#target_link_libraries(ATEM PUBLIC Qt5::Core)
