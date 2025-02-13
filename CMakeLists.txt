cmake_minimum_required(VERSION 3.10)
project(ScratchML C)

set(CMAKE_C_STANDARD 11)

# Enable testing
enable_testing()

# Add include directory
include_directories(include)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)

# Create the main executable from the main.c file
add_executable(main main.c)

# Link the ScratchML library to the main executable
target_link_libraries(main ScratchML)  # This assumes ScratchML is created in src/CMakeLists.txt
