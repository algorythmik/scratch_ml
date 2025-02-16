#include "../include/logistic_regression.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define EPSILON 1e-6

static void test_simple_logistic_regression() {
    // Create a simple dataset for binary classification
    // Points with x < 0 are class 0, points with x > 0 are class 1
    double x_data[] = {-2.0, -1.0, 1.0, 2.0};
    double y_data[] = {0.0, 0.0, 1.0, 1.0};
    
    // Create matrices for X (including bias term) and y
    Matrix X = create_matrix(4, 2);  // 4 samples, 2 features (bias and x)
    Matrix y = matrix_from_array(4, 1, y_data);
    
    // Set up X matrix with bias term (1s) and x values
    for (int i = 0; i < 4; i++) {
        X.data[i * 2] = 1.0;        // bias term
        X.data[i * 2 + 1] = x_data[i];  // x values
    }
    
    // Create and train model
    LogisticRegression *model = LogisticRegression_new(true);
    LogisticRegression_fit(model, &X, &y);
    
    // Test predictions
    Matrix y_pred = create_matrix(4, 1);
    LogisticRegression_predict(model, &X, &y_pred);
    
    // Verify predictions (should correctly classify training points)
    for (int i = 0; i < 2; i++) {  // First two points should be class 0
        assert(y_pred.data[i] < 0.5 && "Should predict class 0");
    }
    for (int i = 2; i < 4; i++) {  // Last two points should be class 1
        assert(y_pred.data[i] > 0.5 && "Should predict class 1");
    }
    
    // Test prediction on new points
    Matrix X_test = create_matrix(2, 2);
    // Set up test points: x = -3 (should be class 0) and x = 3 (should be class 1)
    X_test.data[0] = 1.0; X_test.data[1] = -3.0;  // bias and x for first point
    X_test.data[2] = 1.0; X_test.data[3] = 3.0;   // bias and x for second point
    
    Matrix y_test_pred = create_matrix(2, 1);
    LogisticRegression_predict(model, &X_test, &y_test_pred);
    
    assert(y_test_pred.data[0] < 0.5 && "Should predict class 0 for x = -3");
    assert(y_test_pred.data[1] > 0.5 && "Should predict class 1 for x = 3");
    
    // Clean up
    free_matrix(&X);
    free_matrix(&y);
    free_matrix(&y_pred);
    free_matrix(&X_test);
    free_matrix(&y_test_pred);
    LogisticRegression_free(model);
    
    printf("Simple logistic regression test passed!\n");
}

static void test_multiple_logistic_regression() {
    // Create dataset for binary classification with two features
    // Points inside circle (x²+y²<1) are class 0, outside are class 1
    double x_data[] = {
        0.0, 0.0,    // inside (class 0)
        0.5, 0.0,    // inside (class 0)
        2.0, 0.0,    // outside (class 1)
        0.0, 2.0     // outside (class 1)
    };
    double y_data[] = {0.0, 0.0, 1.0, 1.0};
    
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
    LogisticRegression *model = LogisticRegression_new(true);
    LogisticRegression_fit(model, &X, &y);
    
    // Test predictions on training data
    Matrix y_pred = create_matrix(4, 1);
    LogisticRegression_predict(model, &X, &y_pred);
    
    // Verify predictions
    for (int i = 0; i < 2; i++) {  // First two points should be class 0
        assert(y_pred.data[i] < 0.5 && "Should predict class 0");
    }
    for (int i = 2; i < 4; i++) {  // Last two points should be class 1
        assert(y_pred.data[i] > 0.5 && "Should predict class 1");
    }
    
    // Clean up
    free_matrix(&X);
    free_matrix(&y);
    free_matrix(&y_pred);
    LogisticRegression_free(model);
    
    printf("Multiple logistic regression test passed!\n");
}

static void test_logistic_regression_errors() {
    // Test with invalid inputs
    Matrix X = create_matrix(3, 2);
    Matrix y = create_matrix(2, 1);  // Different number of samples than X
    LogisticRegression *model = LogisticRegression_new();
    
    // Should print error about sample size mismatch
    LogisticRegression_fit(model, &X, &y);
    assert(!model->is_fitted && "Model should not be fitted with invalid input");
    
    // Test prediction without fitting
    Matrix y_pred = create_matrix(3, 1);
    LogisticRegression_predict(model, &X, &y_pred);
    
    // Clean up
    free_matrix(&X);
    free_matrix(&y);
    free_matrix(&y_pred);
    LogisticRegression_free(model);
    
    printf("Logistic regression error handling test passed!\n");
}

int main() {
    printf("Running logistic regression tests...\n");
    
    test_simple_logistic_regression();
    test_multiple_logistic_regression();
    test_logistic_regression_errors();
    
    printf("All logistic regression tests passed!\n");
    return 0;
} 