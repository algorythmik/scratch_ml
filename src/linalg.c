#include "../include/linalg.h"
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Matrix create_matrix(int rows, int cols) {
  Matrix mat = {NULL, rows, cols};
  mat.data = (double *)calloc(rows * cols, sizeof(double));
  if (!mat.data) {
    fprintf(stderr, "Failed to allocate memory for matrix\n");
  }
  return mat;
}
// Free the memory allocated for a matrix
bool free_matrix(Matrix *mat) {
  if (mat && mat->data) {
    free(mat->data);
    mat->data = NULL;

    mat->rows = 0;
    mat->cols = 0;
  }
  return false;
}
bool matmul(const Matrix *restrict A, const Matrix *restrict B, Matrix *restrict C) {
    if (!A || !B || !C) {
        return false;
    }
    
    size_t m = A->rows;
    size_t n = B->cols;
    size_t k = A->cols;
    
    if (k != B->rows) {
        fprintf(stderr, "Matrix dimension mismatch: %d x %d * %d x %d\n", 
                (int)m, (int)k, (int)B->rows, (int)n);
        return false;
    }
    
    if (!C->data) {
        *C = create_matrix(m, n);
        if (!C->data) {
            return false;
        }
    }
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                (int)m, (int)n, (int)k,  // Cast only for CBLAS interface
                1.0, A->data, (int)k, 
                B->data, (int)n, 
                0.0, C->data, (int)n);
    
    return true;
}

void print_matrix(const Matrix *mat) {
    if (!mat || !mat->data) {
        fprintf(stderr, "Invalid matrix\n");
        return;
    }

    printf("Matrix %dx%d:\n", mat->rows, mat->cols);
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            printf("%8.3f ", mat->data[i * mat->cols + j]);
        }
        printf("\n");
    }
}

Matrix matrix_from_array(int rows, int cols, const double values[]) {
    Matrix mat = create_matrix(rows, cols);
    if (!mat.data) {
        return mat;
    }
    
    int size = rows * cols;
    for (int i = 0; i < size; i++) {
        mat.data[i] = values[i];
    }
    
    return mat;
}

bool matrices_equal(const Matrix *A, const Matrix *B, double epsilon) {
    if (!A || !B || !A->data || !B->data) {
        return false;
    }
    
    if (A->rows != B->rows || A->cols != B->cols) {
        return false;
    }
    
    int size = A->rows * A->cols;
    for (int i = 0; i < size; i++) {
        if (fabs(A->data[i] - B->data[i]) >= epsilon) {
            return false;
        }
    }
    
    return true;
}

bool matrix_inverse_raw(int n, double *A, int lda, int *ipiv) {
    int info;
    
    // LU decomposition
    dgetrf_(&n, &n, A, &lda, ipiv, &info);
    if (info != 0) {
        return false;
    }
    
    // Matrix inversion
    int lwork = n * n;
    double *work = (double *)malloc(lwork * sizeof(double));
    dgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
    free(work);
    
    return (info == 0);
}

bool matrix_inverse(const Matrix *A, Matrix *inv_A) {
    if (!A || !inv_A || !A->data || A->rows != A->cols) {
        return false;
    }
    
    int n = A->rows;
    
    // Create or verify output matrix
    if (!inv_A->data || inv_A->rows != n || inv_A->cols != n) {
        *inv_A = create_matrix(n, n);
        if (!inv_A->data) {
            return false;
        }
    }
    
    // Copy input matrix to output (LAPACK routines work in-place)
    memcpy(inv_A->data, A->data, n * n * sizeof(double));
    
    // Allocate pivot indices array
    int *ipiv = (int *)malloc(n * sizeof(int));
    if (!ipiv) {
        return false;
    }
    
    // Perform inversion
    bool success = matrix_inverse_raw(n, inv_A->data, n, ipiv);
    
    free(ipiv);
    return success;
}
