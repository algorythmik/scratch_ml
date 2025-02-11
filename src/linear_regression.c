#include "../include/linear_regression.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

LinearRegression *LinearRegression_new() {
  LinearRegression *model = malloc(sizeof(LinearRegression));
  model->base.fit = LinearRegression_fit;
  model->base.predict = LinearRegression_predict;
  model->base.free = LinearRegression_free;
  model->weights = NULL;
  model->n_features = 0;
  return model;
}

void LinearRegression_fit(void *self, double *X, double *y, int n_samples,
                       int n_features) {
  LinearRegression *model = (LinearRegression *)self;
  model->n_features = n_features;
  model->weights = malloc(n_features * sizeof(double));

  // Simplified: Just setting weights to 1 for now
  for (int i = 0; i < n_features; i++) {
    model->weights[i] = 1.0;
  }

  printf("Model trained with %d features.\n", n_features);
}

void LinearRegression_predict(void *self, double *X, double *y_pred,
                              int n_samples, int n_features) {
  LinearRegression *model = (LinearRegression *)self;
  for (int i = 0; i < n_samples; i++) {
    y_pred[i] = 0;
    for (int j = 0; j < n_features; j++) {
      y_pred[i] += X[i * n_features + j] * model->weights[j];
    }
  }
}

void LinearRegression_free(void *self) {
  LinearRegression *model = (LinearRegression *)self;
  free(model->weights);
  free(model);
}
