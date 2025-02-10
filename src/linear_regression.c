#include "../include/linear_regression.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

LinearRegression *LineaRegression_new() {
  LinearRegression *model = malloc(sizeof(LinearRegression));
  model->base.fit = LinearRegression_fit;
  model->base.predict = LinearRegression_predict;
  model->weights = NULL;
  model->n_features = 0;
  return model;
}

void *LinearRegress_fit(void *self, double *X, double *y, int n_samples,
                        int n_features) {
  LinearRegression *model = (LinearRegression *)self;
  model->n_features = n_features;
  model->weights = malloc(n_features * sizeof(double));

  // Simplified: Just setting weights to 1 for now
  for (int i = 0; i < n_features; i++) {
    model->weights[i] = 1.0;
  }

  printf("Model trained with %d features.\n", n_features);
  return model;
}
