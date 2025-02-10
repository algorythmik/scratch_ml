#include "../include/dataframe.h"
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// create a new DataFrame
DataFrame *df_create(int rows, int cols, char **column_names) {
  DataFrame *df = malloc(sizeof(DataFrame));
  df->rows = rows;
  df->cols = cols;

  // Allocate memory for column names
  df->column_names = malloc(cols * sizeof(char *));
  for (int i = 0; i < cols; i++) {
    df->column_names[i] = strdup(column_names[i]);
  }
  // Allocate memory for data
  df->data = malloc(rows * sizeof(double *));
  for (int i = 0; i < rows; i++) {
    df->data[i] = calloc(cols, sizeof(double));
  }
  return df;
}

// Print DataFrame
void df_print(DataFrame *df) {
  for (int i = 0; i < df->cols; i++) {
    printf("%s\t", df->column_names[i]);
  }
  printf("\n");

  for (int i = 0; i < df->rows; i++) {
    for (int j = 0; j < df->cols; j++) {
      printf("%.2f\t", df->data[i][j]);
    }
    printf("\n");
  }
}

// Free DataFrame memory
void df_free(DataFrame *df) {
  for (int i = 0; i < df->cols; i++) {
    free(df->column_names[i]);
  }
  free(df->column_names);

  for (int i = 0; i < df->rows; i++) {
    free(df->data[i]);
  }
  free(df->data);

  free(df);
}
