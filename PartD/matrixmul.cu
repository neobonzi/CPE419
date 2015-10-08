#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cublas_v2.h>

#define DEF_ROWS 2
#define DEF_COLS 2

// FLOAT will either be a float or double depending on what user decides. (could use a better name)
typedef struct {
  int rows;
  int cols;
  FLOAT *arr;          // array of data
  int mmapFileSize;    // size of file mapped to memory
  char *mmapFileLoc;   // pointer to file mapped to memory
  int size;
} Matrix;

static void HandleError( cudaError_t err,
    const char *file,
    int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
        file, line );
    exit( EXIT_FAILURE );
  }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/**
* This function will take a filename and map its contents into memory for faster access.
*/
int mapFileToMemory(char* fName, Matrix *mat){
  int fd, size;
  char *map;
  struct stat st;

  stat(fName, &st);
  size = st.st_size;      // use stat to get file size
  mat->mmapFileSize = size;

  fd = open(fName, O_RDONLY);
  if (fd == -1) {
    perror("Error, could not open file");
    close(fd);
    exit(1);
  }

  map = (char*) mmap(0, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) {
    perror("Error, could not map file");
    close(fd);
    exit(1);
  }

  mat->mmapFileLoc = map;

  close(fd);
  return  0;
}

/**
* This function will un-map a file that was previously mapped into memory.
*/
int unmapFile(Matrix *mat){
  if (munmap(mat->mmapFileLoc, mat->mmapFileSize) == -1) {
  	perror("Error un-mapping the file");
  	exit(1);
  }

  return 0;
}

/**
* Write results to a file named "results.out"
* Data in mat is stored row-major order.
*/
void writeOutput(Matrix *mat){
  FILE* ofp;
  ofp = fopen("result.out","w");
  if(ofp == NULL) {
    perror("Could not open result.out to write results");
    exit(1);
  }

  int i,j;
  for(i = 0; i < mat->rows; i++){
    for(j = 0; j < mat->cols; j++){
      fprintf(ofp, "%.2f ", mat->arr[i * mat->cols + j]);
    }
    // print newline for all rows
    fprintf(ofp, "\n");
  }

  // close output file pointer
  fclose(ofp);
}

/**
* Write results to a file named "results.out"
* Data in mat is stored column-major order.
*/
void writeOutputColMajor(Matrix *mat){
  FILE* ofp;
  ofp = fopen("result.out","w");
  if(ofp == NULL) {
    perror("Could not open result.out to write results");
    exit(1);
  }

  int i,j;
  for(i = 0; i < mat->rows; i++){
    for(j = 0; j < mat->cols; j++){
      fprintf(ofp, "%.2f ", mat->arr[i + mat->rows * j]);
    }
    // print newline for all rows
    fprintf(ofp, "\n");
  }

  // close output file pointer
  fclose(ofp);
}

/**
* Print the contents of matrix to stdout for debugging
*/
void printMatrix(Matrix *mat) {
  int i,j;
  for(i = 0; i < mat->rows; i++){
    for(j = 0; j < mat->cols; j++){
      printf("%.2f ", mat->arr[i * mat->cols + j]);
    }
    // print newline for all rows
    printf("\n");
  }
}

/**
* Print the contents of matrix to stdout for debugging
*/
void printMatrixAll(Matrix *mat) {
  int i;
  for(i = 0; i < mat->rows * mat->cols; i++){
    printf("%.2f ", mat->arr[i]);
  }
}

/**
* Initialize an array in memory to hold matrix data.
*/
void initMatrixArray(Matrix *mat, int initRows, int initCols) {
  mat->rows = initRows;
  mat->cols = initCols;
  mat->size = mat->rows * mat->cols;

  FLOAT *newArray = (FLOAT *) malloc(sizeof(FLOAT) * mat->rows * mat->cols);

  if (newArray == NULL) {
    perror("Error, couldn't allocate space for array");
    exit(1);
  }

  // set pointer to new array in memory
  mat->arr = newArray;
}

/**
* If array holding Matrix data is not big enough create a new one twice as big.
* Copy old array data to new array, and free old array from memory.
*/
void doubleArraySize(Matrix *mat) {
  // malloc new array, double the size of previous array
  FLOAT *newArray = (FLOAT *) malloc(sizeof(FLOAT) * mat->size * 2);

  if (newArray == NULL) {
    perror("Error, couldn't allocate space for array\n");
    exit(1);
  }

  // copy old array to newArray
  newArray = (FLOAT*) memcpy(newArray, mat->arr, sizeof(FLOAT) * mat->size);

  // update size of array
  mat->size *= 2;

  // free the old array
  free(mat->arr);

  // set pointer to new array in memory
  mat->arr = newArray;
}

/**
* Read values from memory mapped location into an array for processing.
* This function reads in the characters and converts them to floats or doubles
* before storing in the array.
*/
void storeMatrixToArray(Matrix *mat){
  int mmapIdx = 0, bfrIdx = 0, arrIdx = 0, numRows = 0, numCols = 0, countCols = 1;
  char buffer[101];    // buffer to hold FLOAT up to 100 digits long

  for(mmapIdx = 0; mmapIdx <= mat->mmapFileSize; mmapIdx++) {
    if(mat->mmapFileLoc[mmapIdx] == ' '){ // found a number, store into Matrix
      // null-terminate the buffer so it can be passed to atof()
      buffer[bfrIdx] = '\0';

      // if array is full, double size
      if (arrIdx == mat->size) {
        doubleArraySize(mat);
      }

      // convert char buffer to FLOAT and store in Matrix array
      mat->arr[arrIdx++] = (FLOAT) atof(buffer);

      // only count number of columns until reach first newline
      if (countCols) numCols++;

      // clear buffer, reset buffer index to 0
      memset(buffer, '\0', bfrIdx);
      bfrIdx = 0;
    } else if(mat->mmapFileLoc[mmapIdx] == '\n') {
      if (countCols) {
        mat->cols = numCols;
        countCols = 0;    // disable TILE_WIDTH - (mat->cols % TILE_WIDTH),counting columns
      }
      numRows++;
    }

    /* grab a character at each loop iteration and store into buffer[] to conv to FLOAT */
    buffer[bfrIdx++] = mat->mmapFileLoc[mmapIdx];
  }

  mat->rows = numRows;
}

int errorCheckMatrices(Matrix *mat1, Matrix *mat2){
  // check that matrix1 is same size as matrix2
  if(mat1->cols != mat2->rows){
    printf("Error: matrices are not compatible size\n");
    exit(1);
  }

  return 0;
}

/**
* In order for faster matrix processing, this function will take an array
* of matrix data in row-major order and convert it to column-major order.
*/
void convValuesColMajor(Matrix *p) {
  FLOAT *newArray = (FLOAT *) malloc(sizeof(FLOAT) * p->rows * p->cols);

  if (newArray == NULL) {
    perror("Error, couldn't allocate space for array");
    exit(1);
  }

  int iterator = 0;
  int i, j, idx;
  for(i = 0; i < p->rows; i++){
    idx = i;
    for(j = 0; j < p->cols; j++){
      newArray[idx] = p->arr[iterator++];
      idx += p->rows;
    }
  }

  // free the old array
  free(p->arr);

  // set pointer to new array
  p->arr = newArray;
}

/**
 * This function will multiply using the cuBLAS library.
 */
void cublasMatMul(FLOAT *Md, FLOAT *Nd, FLOAT *Pd, Matrix *mat1, Matrix *mat2, Matrix *mat3) {

  float a = 1.0;
  float b = 0.0;
  float *alpha = &a;
  float *beta = &b;
  
  int lda = mat1->rows;
  int ldb = mat2->rows;
  int ldc = mat3->rows;
  
  // cublas structures
  cublasHandle_t h;
  
  // link handle
  cublasCreate(&h);

  // matrix multiply with cuBLAS.
  // Note: CUBLAS_OP_N signfifies no transposition for matrix1 and matrix2
  // IMPORTANT: to use cuBLAS matrices must be in column-major order
  cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, mat1->rows, mat2->cols, mat1->cols, 
              alpha, Md, lda, Nd, ldb, beta, Pd, ldc);
              
  cublasDestroy(h);
}

void matrixMulOnDevice(Matrix *mat1, Matrix *mat2, Matrix *mat3){
  int size;
  FLOAT *Md, *Nd, *Pd;

  // Allocate space and send data for Matrix1 to GPU
  size = mat1->rows * mat1->cols * sizeof(FLOAT);
  HANDLE_ERROR( cudaMalloc(&Md, size) );
  HANDLE_ERROR( cudaMemcpy(Md, mat1->arr, size, cudaMemcpyHostToDevice) );

  // Allocate space and send data for Matrix2 to GPU
  size = mat2->rows * mat2->cols * sizeof(FLOAT);
  HANDLE_ERROR( cudaMalloc(&Nd, size) );
  HANDLE_ERROR( cudaMemcpy(Nd, mat2->arr, size, cudaMemcpyHostToDevice) );

  // Allocate space for Matrix3 on GPU
  size = mat3->rows * mat3->cols * sizeof(FLOAT);
  HANDLE_ERROR( cudaMalloc(&Pd, size) );

  // Compute matrix with cuBLAS library
  cublasMatMul(Md, Nd, Pd, mat1, mat2, mat3);

  // Copy results back to CPU
  HANDLE_ERROR( cudaMemcpy(mat3->arr, Pd, size, cudaMemcpyDeviceToHost) );
  
  // Free GPU memory
  HANDLE_ERROR( cudaFree(Md) );
  HANDLE_ERROR( cudaFree(Nd) );
  HANDLE_ERROR( cudaFree(Pd) );
}

int main( int argc, char **argv ) {
  // exit if not enough arguments
  if (argc != 3) {
    printf("Error: incorrect amount of arguments");
    exit(1);
  }

  // Setup matrix1
  Matrix m1;
  Matrix *pMat1 = &m1;
  mapFileToMemory(argv[1], pMat1);
  initMatrixArray(pMat1, DEF_ROWS, DEF_COLS);
  storeMatrixToArray(pMat1);
  convValuesColMajor(pMat1);
  unmapFile(pMat1);

  // Setup matrix2
  Matrix m2;
  Matrix *pMat2 = &m2;
  mapFileToMemory(argv[2], pMat2);
  initMatrixArray(pMat2, DEF_ROWS, DEF_COLS);
  storeMatrixToArray(pMat2);
  convValuesColMajor(pMat2);
  unmapFile(pMat2);

  // check for errors
  errorCheckMatrices(pMat1, pMat2);
  
  // Matrix3 setup and compute
  Matrix m3;
  Matrix *pMat3 = &m3;
  initMatrixArray(pMat3, pMat1->rows, pMat2->cols);
  matrixMulOnDevice(pMat1, pMat2, pMat3);
  writeOutputColMajor(pMat3);
  
  // Free allocated memory
  free(pMat1->arr);
  free(pMat2->arr);
  free(pMat3->arr);

  return 0;
}
