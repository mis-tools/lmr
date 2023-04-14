cmake_minimum_required(VERSION 2.8)
 
project(lmr)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

find_package(OpenMP)
if ( ${OPENMP_FOUND} )
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()
 
add_executable(lmr src/lmr.cxx)
 
target_link_libraries(lmr ${ITK_LIBRARIES})

INSTALL(TARGETS lmr DESTINATION bin/)
