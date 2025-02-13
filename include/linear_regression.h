#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "linalg.h"
#include "base.h"

typedef struct {
    MLBase base;
    Matrix weights;  // Changed from double* to Matrix
    bool is_fitted;
} LinearRegression;

LinearRegression *LinearRegression_new(void);
void LinearRegression_fit(void *self, const Matrix *X, const Matrix *y);
void LinearRegression_predict(void *self, const Matrix *X, Matrix *y_pred);
void LinearRegression_free(void *self);

#endif // !LINEAR_REGRESSION_H
