#include "../include/linear_regression.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define EPSILON 1e-6

static void test_simple_linear_regression() {
    // Create a simple dataset: y = 2x + 1
    double x_data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y_data[] = {3.0, 5.0, 7.0, 9.0, 11.0};
    
    // Create matrices for X (including bias term) and y
    Matrix X = create_matrix(5, 2);  // 5 samples, 2 features (bias and x)
    Matrix y = matrix_from_array(5, 1, y_data);
    
    // Set up X matrix with bias term (1s) and x values
    for (int i = 0; i < 5; i++) {
        X.data[i * 2] = 1.0;        // bias term
        X.data[i * 2 + 1] = x_data[i];  // x values
    }
    
    // Create and train model
    LinearRegression *model = LinearRegression_new();
    LinearRegression_fit(model, &X, &y);
    
    // Check if weights are close to expected values (β₀=1, β₁=2)
    assert(fabs(model->weights.data[0] - 1.0) < EPSILON && "Incorrect bias term");
    assert(fabs(model->weights.data[1] - 2.0) < EPSILON && "Incorrect slope");
    
    // Test predictions
    Matrix y_pred = create_matrix(5, 1);
    LinearRegression_predict(model, &X, &y_pred);
    
    // Verify predictions
    for (int i = 0; i < 5; i++) {
        assert(fabs(y_pred.data[i] - y_data[i]) < EPSILON && 
               "Prediction doesn't match expected value");
    }
    
    // Clean up
    free_matrix(&X);
    free_matrix(&y);
    free_matrix(&y_pred);
    LinearRegression_free(model);
    
    printf("Simple linear regression test passed!\n");
}

static void test_multiple_linear_regression() {
    // Create dataset: y = 1 + 2x₁ + 3x₂
    double x_data[] = {
        1.0, 2.0,  // sample 1
        2.0, 1.0,  // sample 2
        3.0, 3.0,  // sample 3
        4.0, 2.0   // sample 4
    };
    double y_data[] = {9.0, 8.0, 16.0, 15.0};  // Corresponding y values
    
    // Create matrices
    Matrix X = create_matrix(4, 3);  // 4 samples, 3 features (bias, x₁, x₂)
    Matrix y = matrix_from_array(4, 1, y_data);
    
    // Set up X matrix with bias term and features
    for (int i = 0; i < 4; i++) {
        X.data[i * 3] = 1.0;            // bias term
        X.data[i * 3 + 1] = x_data[i * 2];    // x₁
        X.data[i * 3 + 2] = x_data[i * 2 + 1];  // x₂
    }
    
    // Create and train model
    LinearRegression *model = LinearRegression_new();
    LinearRegression_fit(model, &X, &y);
    
    // Check if weights are close to expected values (β₀=1, β₁=2, β₂=3)
    assert(fabs(model->weights.data[0] - 1.0) < EPSILON && "Incorrect bias term");
    assert(fabs(model->weights.data[1] - 2.0) < EPSILON && "Incorrect coefficient for x₁");
    assert(fabs(model->weights.data[2] - 3.0) < EPSILON && "Incorrect coefficient for x₂");
    
    // Test predictions
    Matrix y_pred = create_matrix(4, 1);
    LinearRegression_predict(model, &X, &y_pred);
    
    // Verify predictions
    for (int i = 0; i < 4; i++) {
        assert(fabs(y_pred.data[i] - y_data[i]) < EPSILON && 
               "Prediction doesn't match expected value");
    }
    
    // Clean up
    free_matrix(&X);
    free_matrix(&y);
    free_matrix(&y_pred);
    LinearRegression_free(model);
    
    printf("Multiple linear regression test passed!\n");
}

static void test_linear_regression_errors() {
    // Test with invalid inputs
    Matrix X = create_matrix(3, 2);
    Matrix y = create_matrix(2, 1);  // Different number of samples than X
    LinearRegression *model = LinearRegression_new();
    
    // Should print error about sample size mismatch
    LinearRegression_fit(model, &X, &y);
    assert(!model->is_fitted && "Model should not be fitted with invalid input");
    
    // Test prediction without fitting
    Matrix y_pred = create_matrix(3, 1);
    LinearRegression_predict(model, &X, &y_pred);
    
    // Clean up
    free_matrix(&X);
    free_matrix(&y);
    free_matrix(&y_pred);
    LinearRegression_free(model);
    
    printf("Linear regression error handling test passed!\n");
}

int main() {
    printf("Running linear regression tests...\n");
    
    test_simple_linear_regression();
    test_multiple_linear_regression();
    test_linear_regression_errors();
    
    printf("All linear regression tests passed!\n");
    return 0;
} 