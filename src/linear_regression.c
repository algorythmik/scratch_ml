#include "../include/linear_regression.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

LinearRegression *LinearRegression_new() {
    LinearRegression *model = malloc(sizeof(LinearRegression));
    if (!model) {
        return NULL;
    }
    
    model->base.fit = LinearRegression_fit;
    model->base.predict = LinearRegression_predict;
    model->base.free = LinearRegression_free;
    model->weights.data = NULL;  // Initialize empty matrix
    model->is_fitted = false;
    
    return model;
}

void LinearRegression_fit(void *self, const Matrix *X, const Matrix *y) {
    LinearRegression *model = (LinearRegression *)self;
    
    if (!X || !y || !X->data || !y->data) {
        fprintf(stderr, "Invalid input matrices\n");
        return;
    }
    
    if (X->rows != y->rows) {
        fprintf(stderr, "Number of samples in X and y must match\n");
        return;
    }
    
    // Compute (X^T * X)^(-1) * X^T * y
    
    // Step 1: Calculate X^T
    Matrix X_transpose = create_matrix(X->cols, X->rows);
    for (int i = 0; i < X->rows; i++) {
        for (int j = 0; j < X->cols; j++) {
            X_transpose.data[j * X->rows + i] = X->data[i * X->cols + j];
        }
    }
    
    // Step 2: Calculate X^T * X
    Matrix XtX = create_matrix(X->cols, X->cols);
    matmul(&X_transpose, X, &XtX);
    
    // Step 3: Calculate (X^T * X)^(-1)
    Matrix XtX_inv = create_matrix(X->cols, X->cols);
    if (!matrix_inverse(&XtX, &XtX_inv)) {
        fprintf(stderr, "Failed to compute matrix inverse\n");
        goto cleanup;
    }
    
    // Step 4: Calculate X^T * y
    Matrix Xty = create_matrix(X->cols, 1);
    matmul(&X_transpose, y, &Xty);
    
    // Step 5: Calculate (X^T * X)^(-1) * X^T * y
    model->weights = create_matrix(X->cols, 1);
    matmul(&XtX_inv, &Xty, &(model->weights));
    
    model->is_fitted = true;
    printf("Model trained with %d features.\n", X->cols);
    
cleanup:
    free_matrix(&X_transpose);
    free_matrix(&XtX);
    free_matrix(&XtX_inv);
    free_matrix(&Xty);
}

void LinearRegression_predict(void *self, const Matrix *X, Matrix *y_pred) {
    LinearRegression *model = (LinearRegression *)self;
    
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
    
    // Compute X * weights
    matmul(X, &(model->weights), y_pred);
}

void LinearRegression_free(void *self) {
    LinearRegression *model = (LinearRegression *)self;
    if (model) {
        free_matrix(&(model->weights));
        free(model);
    }
}
