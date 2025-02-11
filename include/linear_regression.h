#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H
#include "../src/base.c"
typedef struct {
  MLBase base;
  double *weights;
  int n_features;
} LinearRegression;

LinearRegression *LinearRegression_new();
void LinearRegression_fit(void *self, double *X, double *y, int n_samples,
                          int n_feaures);
void LinearRegression_predict(void *self, double *X, double *y, int n_samples,
                              int n_feaures);
void LinearRegression_free(void *self);
#endif // !LINEAR_REGRESSION_H
