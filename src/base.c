#ifndef BASE_H
#define BASE_H
typedef struct {
  void (*fit)(void *self, double *X, double *y, int n_samples, int n_features);
  void (*predict)(void *self, double *X,  int n_samples, int features);
} MLBase;
#endif /* ifndef BASE_H                                                        \
        */
