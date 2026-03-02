cmake_minimum_required(VERSION 3.15)
project(cboard)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(cboard board.cpp)
target_include_directories(cboard PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(cboard PRIVATE -O3 -march=native)
# Static-link MinGW runtime so the .pyd has no external GCC/STL DLL deps
target_link_options(cboard PRIVATE -static-libgcc -static-libstdc++)
# Install directly into chessbot/ so `from chessbot.cboard import Position` works
install(TARGETS cboard DESTINATION ${CMAKE_INSTALL_PREFIX})
