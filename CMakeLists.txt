cmake_minimum_required(VERSION 3.4...3.18)
project(qmekf)

set (CMAKE_CXX_STANDARD 17)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# The main library, to be included anywhere you like
add_library(qmekf INTERFACE)
target_include_directories(qmekf INTERFACE include)

target_link_libraries(qmekf INTERFACE Eigen3::Eigen)

add_executable(qmekf_main src/main.cpp)
target_link_libraries(qmekf_main qmekf)