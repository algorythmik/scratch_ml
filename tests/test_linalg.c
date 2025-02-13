#include "../include/linalg.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define EPSILON 1e-10

static bool doubles_equal(double a, double b) {
    return fabs(a - b) < EPSILON;
}

static void test_matmul_2x2() {
    // Initialize matrices with values
    double values_a[] = {1.0, 2.0, 3.0, 4.0};
    double values_b[] = {5.0, 6.0, 7.0, 8.0};
    double expected_values[] = {19.0, 22.0, 43.0, 50.0};
    
    Matrix A = matrix_from_array(2, 2, values_a);
    Matrix B = matrix_from_array(2, 2, values_b);
    Matrix C = create_matrix(2, 2);
    Matrix expected = matrix_from_array(2, 2, expected_values);

    // Perform multiplication
    bool success = matmul(&A, &B, &C);
    assert(success && "Matrix multiplication failed");

    // Check result
    assert(matrices_equal(&C, &expected, EPSILON) && "Matrix multiplication result incorrect");

    // Clean up
    free_matrix(&A);
    free_matrix(&B);
    free_matrix(&C);
    free_matrix(&expected);
    
    printf("2x2 matrix multiplication test passed!\n");
}

static void test_matmul_invalid() {
    // Test with incompatible dimensions
    Matrix A = create_matrix(2, 3);
    Matrix B = create_matrix(2, 2);
    Matrix C = create_matrix(2, 2);

    bool success = matmul(&A, &B, &C);
    assert(!success && "Should fail with incompatible dimensions");

    free_matrix(&A);
    free_matrix(&B);
    free_matrix(&C);

    printf("Invalid matrix multiplication test passed!\n");
}

static void test_matrix_from_array() {
    double values[] = {1.0, 2.0, 3.0, 4.0};
    Matrix mat = matrix_from_array(2, 2, values);
    
    assert(mat.rows == 2);
    assert(mat.cols == 2);
    assert(doubles_equal(mat.data[0], 1.0));
    assert(doubles_equal(mat.data[1], 2.0));
    assert(doubles_equal(mat.data[2], 3.0));
    assert(doubles_equal(mat.data[3], 4.0));
    
    free_matrix(&mat);
    printf("Matrix from array test passed!\n");
}

static void test_matrices_equal() {
    // Test equal matrices
    double values1[] = {1.0, 2.0, 3.0, 4.0};
    double values2[] = {1.0, 2.0, 3.0, 4.0};
    Matrix A = matrix_from_array(2, 2, values1);
    Matrix B = matrix_from_array(2, 2, values2);
    
    assert(matrices_equal(&A, &B, EPSILON) && "Equal matrices should compare as equal");
    
    // Test matrices with small differences (within epsilon)
    B.data[0] += EPSILON / 2;
    assert(matrices_equal(&A, &B, EPSILON) && "Matrices within epsilon should compare as equal");
    
    // Test matrices with differences larger than epsilon
    B.data[0] = A.data[0] + EPSILON * 2;
    assert(!matrices_equal(&A, &B, EPSILON) && "Matrices outside epsilon should not compare as equal");
    
    // Test matrices with different dimensions
    Matrix C = create_matrix(2, 3);
    assert(!matrices_equal(&A, &C, EPSILON) && "Matrices with different dimensions should not compare as equal");
    
    free_matrix(&A);
    free_matrix(&B);
    free_matrix(&C);
    
    printf("Matrix equality test passed!\n");
}

static void test_matrix_inverse() {
    // Test matrix inversion with a simple 2x2 matrix
    double values[] = {4.0, 7.0, 
                      2.0, 6.0};
    double expected[] = {0.6, -0.7,
                        -0.2, 0.4};
    
    Matrix A = matrix_from_array(2, 2, values);
    Matrix inv_A = create_matrix(2, 2);
    Matrix expected_inv = matrix_from_array(2, 2, expected);
    
    // Perform inversion
    bool success = matrix_inverse(&A, &inv_A);
    assert(success && "Matrix inversion failed");
    
    // Check result
    assert(matrices_equal(&inv_A, &expected_inv, 1e-6) && 
           "Matrix inverse result incorrect");
    
    // Verify A * A^(-1) = I
    Matrix I = create_matrix(2, 2);
    matmul(&A, &inv_A, &I);
    
    double identity[] = {1.0, 0.0,
                        0.0, 1.0};
    Matrix expected_I = matrix_from_array(2, 2, identity);
    assert(matrices_equal(&I, &expected_I, 1e-6) && 
           "A * A^(-1) should equal identity matrix");
    
    // Clean up
    free_matrix(&A);
    free_matrix(&inv_A);
    free_matrix(&expected_inv);
    free_matrix(&I);
    free_matrix(&expected_I);
    
    printf("Matrix inversion test passed!\n");
}

int main() {
    printf("Running matrix multiplication tests...\n");
    
    test_matmul_2x2();
    test_matmul_invalid();
    test_matrix_from_array();
    test_matrices_equal();
    test_matrix_inverse();
    
    printf("All tests passed!\n");
    return 0;
} 