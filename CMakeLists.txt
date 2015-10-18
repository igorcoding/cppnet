cmake_minimum_required(VERSION 3.2)
project(CPPNET)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

include_directories(include)

set(HEADERS
        include/Net.h
        include/NetConf.h
        include/TrainData.h
        include/TrainExample.h
        include/Layer.h
        include/InputLayer.h
        include/ForwardLayer.h
        include/RecurrentLayer.h
        include/StreamLayer.h
        include/OutputLayer.h

        include/exceptions/CppNetException.h
        include/exceptions/EndOfDataException.h
)

set(SOURCE_FILES
        src/Net.cc
        src/InputLayer.cc
        src/ForwardLayer.cc
        src/RecurrentLayer.cc
        src/OutputLayer.cc
)

add_library(cppnet ${SOURCE_FILES} ${HEADERS})
target_link_libraries(cppnet ${ARMADILLO_LIBRARIES})

add_executable(ex_main ex/main.cc)
target_link_libraries(ex_main cppnet)

add_executable(ex_xor ex/xor.cc)
target_link_libraries(ex_xor cppnet)

add_executable(ex_test ex/test.cc)
target_link_libraries(ex_test cppnet)
