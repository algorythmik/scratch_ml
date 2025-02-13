enable_testing()

# Add test executable
add_executable(test_linalg test_linalg.c)

# Link with the main library
target_link_libraries(test_linalg ScratchML)

# Add test to CTest
add_test(NAME test_linalg COMMAND test_linalg) 