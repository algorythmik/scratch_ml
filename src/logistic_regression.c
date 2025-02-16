#include "../include/logistic_regression.h"
#include "../include/linalg.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Sigmoid function
static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

LogisticRegression* LogisticRegression_new(bool fit_intercept) {
    LogisticRegression* model = malloc(sizeof(LogisticRegression));
    if (!model) {
        return NULL;
    }
    model->fit_intercept = fit_intercept;
    model->base.fit = LogisticRegression_fit;
    model->base.predict = LogisticRegression_predict;
    model->base.free = LogisticRegression_free;
    model->weights.data = NULL;
    model->is_fitted = false;
    
    return model;
}

void LogisticRegression_fit(void* self, const Matrix* X, const Matrix* y) {
    LogisticRegression* model = (LogisticRegression*)self;
    if (model->fit_intercept) {
        Matrix intercept = create_matrix(X->rows, 1);
        Matrix X_with_intercept = matrix_concat(X, &intercept, 1);
        X = &X_with_intercept;
    }
    if (!X || !y || !X->data || !y->data) {
        fprintf(stderr, "Invalid input matrices\n");
        return;
    }
    
    if (X->rows != y->rows) {
        fprintf(stderr, "Number of samples in X and y must match\n");
        return;
    }

    // Initialize weights to zeros
    if (model->weights.data) {
        free_matrix(&model->weights);
    }
    model->weights = create_matrix(X->cols, 1);
    
    // Gradient descent parameters
    const int max_iter = 1000;
    const double learning_rate = 0.1;
    const double epsilon = 1e-6;
    
    Matrix h = create_matrix(X->rows, 1);  // Predictions
    Matrix error = create_matrix(X->rows, 1);  // Error term
    Matrix X_transpose = create_matrix(X->cols, X->rows);
    Matrix gradient = create_matrix(X->cols, 1);
    
    // Calculate X transpose once
    for (int i = 0; i < X->rows; i++) {
        for (int j = 0; j < X->cols; j++) {
            X_transpose.data[j * X->rows + i] = X->data[i * X->cols + j];
        }
    }
    
    // Gradient descent
    for (int iter = 0; iter < max_iter; iter++) {
        // Calculate predictions h = sigmoid(X * weights)
        matmul(X, &model->weights, &h);
        for (int i = 0; i < h.rows; i++) {
            h.data[i] = sigmoid(h.data[i]);
        }
        
        // Calculate error = h - y
        for (int i = 0; i < error.rows; i++) {
            error.data[i] = h.data[i] - y->data[i];
        }
        
        // Calculate gradient = X^T * error
        matmul(&X_transpose, &error, &gradient);
        
        // Update weights
        double max_change = 0.0;
        for (int i = 0; i < model->weights.rows; i++) {
            double change = learning_rate * gradient.data[i];
            model->weights.data[i] -= change;
            max_change = fmax(max_change, fabs(change));
        }
        
        // Check convergence
        if (max_change < epsilon) {
            break;
        }
    }
    
    // Clean up
    free_matrix(&h);
    free_matrix(&error);
    free_matrix(&X_transpose);
    free_matrix(&gradient);
    
    model->is_fitted = true;
}

void LogisticRegression_predict(void* self, const Matrix* X, Matrix* y_pred) {
    LogisticRegression* model = (LogisticRegression*)self;
    
    if (!model->is_fitted) {
        fprintf(stderr, "Model not fitted yet\n");
        return;
    }
    
    if (!X || !X->data || !y_pred) {
        fprintf(stderr, "Invalid input matrices\n");
        return;
    }
    
    if (X->cols != model->weights.rows) {
        fprintf(stderr, "Number of features in X must match model\n");
        return;
    }
    
    // Compute X * weights and apply sigmoid
    matmul(X, &model->weights, y_pred);
    for (int i = 0; i < y_pred->rows; i++) {
        y_pred->data[i] = sigmoid(y_pred->data[i]);
    }
}

void LogisticRegression_free(void* self) {
    LogisticRegression* model = (LogisticRegression*)self;
    if (model) {
        free_matrix(&model->weights);
        free(model);
    }
}