cmake_minimum_required(VERSION 3.5)
project(yolov8)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(target_bbox_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(CUDA REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/src/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)


file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
cuda_add_executable(yolov8_det ${PROJECT_SOURCE_DIR}/src/yolov8_det.cpp ${SRCS})

ament_target_dependencies(yolov8_det rclcpp sensor_msgs OpenCV cv_bridge target_bbox_msgs CUDA)

target_link_libraries(yolov8_det nvinfer)
target_link_libraries(yolov8_det cudart)
target_link_libraries(yolov8_det myplugins)
target_link_libraries(yolov8_det ${OpenCV_LIBS})

install(DIRECTORY launch weights
  DESTINATION share/${PROJECT_NAME})

install(TARGETS
  yolov8_det
  DESTINATION lib/${PROJECT_NAME}
)
