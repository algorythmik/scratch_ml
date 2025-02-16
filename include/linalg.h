#ifndef LINALG_H
#define LINALG_H

#include <stdbool.h>

typedef struct {
  double *data;
  int rows;
  int cols;
} Matrix;

Matrix create_matrix(int rows, int cols);
bool free_matrix(Matrix *mat);

bool matmul(const Matrix *A, const Matrix *B, Matrix *C);

void print_matrix(const Matrix *mat);

Matrix matrix_from_array(int rows, int cols, const double values[]);

bool matrices_equal(const Matrix *A, const Matrix *B, double epsilon);

bool matrix_inverse(const Matrix *A, Matrix *inv_A);
bool matrix_inverse_raw(int n, double *A, int lda, int *ipiv);

// axis: 0 for vertical stacking, 1 for horizontal stacking
Matrix matrix_concat(const Matrix *A, const Matrix *B, int axis);

#endif // !LINALG_H
