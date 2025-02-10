#ifndef DATAFRAME_H
#define DATAFRAME_H

#include <stdio.h>
#include <stdlib.h>

// DataFrame structure
typedef struct {
  int rows;
  int cols;
  char **column_names;
  double **data;
} DataFrame;

// Function prototypes
DataFrame *df_create(int rows, int cols, char **column_names);
void df_print(DataFrame *df);
void df_free(DataFrame *df);

#endif // DATAFRAME_H
