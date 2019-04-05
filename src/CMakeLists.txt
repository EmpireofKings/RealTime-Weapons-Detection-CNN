

cmake_minimum_required(VERSION 2.8.12)

project(gun_detection)

#Put the path to the uncompressor dlib here.
#add_subdirectory(/home/james/Development/dlib-19.16/dlib dlib_build)

add_executable(gun_trainer trainer_gun.cpp)
add_executable(gun_tester tester_gun.cpp)

target_link_libraries(gun_trainer dlib openblas)
target_link_libraries(gun_tester dlib openblas opencv_core opencv_videoio opencv_imgproc)

