enable_testing()

# Add test executables
add_executable(test_linalg test_linalg.c)
add_executable(test_linear_regression test_linear_regression.c)

# Link with the main library
target_link_libraries(test_linalg ScratchML)
target_link_libraries(test_linear_regression ScratchML)

# Add tests to CTest
add_test(NAME test_linalg COMMAND test_linalg)
add_test(NAME test_linear_regression COMMAND test_linear_regression) 