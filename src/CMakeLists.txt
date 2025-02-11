# Create a static library for ScratchML from the source files
add_library(ScratchML STATIC
    base.c
    # dataframe.c
    linear_regression.c
)

# Optionally link any additional libraries or set compile flags here
# target_link_libraries(ScratchML some_other_lib)
