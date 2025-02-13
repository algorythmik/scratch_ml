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

// Add new function to initialize matrix from array
Matrix matrix_from_array(int rows, int cols, const double values[]);

// Add new function to compare matrices
bool matrices_equal(const Matrix *A, const Matrix *B, double epsilon);

#endif // !LINALG_H
