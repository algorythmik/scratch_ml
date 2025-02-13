#include "include/linear_regression.h"
#include <stdio.h>

int main() {
    int n_samples = 3;
    int n_features = 2;
    
    // Create X matrix (3x2) with sample data
    double X_data[] = {1, 2, 3, 4, 5, 6};
    Matrix X = matrix_from_array(n_samples, n_features, X_data);

    // Create y vector (3x1) with target values
    double y_data[] = {2, 3, 4};
    Matrix y = matrix_from_array(n_samples, 1, y_data);
    
    // Create matrix for predictions
    Matrix y_pred = create_matrix(n_samples, 1);

    // Create and train model
    LinearRegression *lr = LinearRegression_new();
    LinearRegression_fit(lr, &X, &y);
    LinearRegression_predict(lr, &X, &y_pred);

    printf("Predictions:\n");
    print_matrix(&y_pred);

    // Clean up
    free_matrix(&X);
    free_matrix(&y);
    free_matrix(&y_pred);
    LinearRegression_free(lr);
    
    return 0;
}
