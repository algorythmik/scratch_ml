#include "include/linear_regression.h"
#include <stdio.h>

int main() {
  int n_samples = 3;
  int n_features = 2;
  double X[] = {1, 2, 3, 4, 5, 6}; // 3x2 matrix
  double y[] = {2, 3, 4};
  double y_pred[3];

  LinearRegression *lr = LinearRegression_new();
  lr->base.fit(lr, X, y, n_samples, n_features);
  lr->base.predict(lr, X, y_pred, n_samples, n_features);

  printf("Predictions:\n");
  for (int i = 0; i < n_samples; i++) {
    printf("%f\n", y_pred[i]);
  }

  lr->base.free(lr);
  return 0;
}
