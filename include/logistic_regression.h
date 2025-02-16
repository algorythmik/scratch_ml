#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "base.h"
#include "linalg.h"
#include <stdbool.h>

typedef struct {
    MLBase base;
    Matrix weights;
    bool is_fitted;
    bool fit_intercept;
} LogisticRegression;

// Constructor and destructor
LogisticRegression* LogisticRegression_new(bool fit_intercept);
void LogisticRegression_free(void* self);

// Core methods
void LogisticRegression_fit(void* self, const Matrix* X, const Matrix* y);
void LogisticRegression_predict(void* self, const Matrix* X, Matrix* y_pred);

#endif // LOGISTIC_REGRESSION_H 