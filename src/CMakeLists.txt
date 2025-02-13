# Create a static library for ScratchML from the source files
add_library(ScratchML STATIC
    linalg.c
    # dataframe.c
    linear_regression.c
)

# Link with Accelerate framework on macOS
if(APPLE)
    target_link_libraries(ScratchML "-framework Accelerate")
endif()

# Optionally link any additional libraries or set compile flags here
# target_link_libraries(ScratchML some_other_lib)
