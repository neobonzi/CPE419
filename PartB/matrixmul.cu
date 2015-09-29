#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/mman.h>

#define DEF_ROWS 2
#define DEF_COLS 2

// FLOAT will either be a float or double depending on what user decides. (could use a better name)
typedef struct {
  int rows;
  int cols;
  FLOAT *arr;          // array of data
  int mmapFileSize;    // size of file mapped to memory
  char *mmapFileLoc;   // pointer to file mapped to memory
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
      fprintf(ofp, "%.2f ", mat->arr[i*mat->cols+j]);
    }
    // print newline for all rows
    fprintf(ofp, "\n");
  }
  
  // close output file pointer
  fclose(ofp);
}

/**
* Initialize an array in memory to hold matrix data.
*/
void initMatrixArray(Matrix *mat, int initRows, int initCols) {
  mat->rows = initRows;
  mat->cols = initCols;
    
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
  FLOAT *newArray = (FLOAT *) malloc(sizeof(FLOAT) * mat->rows * mat->cols * 2);
  
  if (newArray == NULL) {
    perror("Error, couldn't allocate space for array\n");
    exit(1);
  }
  
  // copy old array to newArray
  newArray = (FLOAT*) memcpy(newArray, mat->arr, sizeof(FLOAT) * mat->rows * mat->cols);
  
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
  int matSize = mat->rows * mat->cols;
  char buffer[101];    // buffer to hold FLOAT up to 100 digits long
  
  for(mmapIdx = 0; mmapIdx <= mat->mmapFileSize; mmapIdx++) {
    if(mat->mmapFileLoc[mmapIdx] == ' '){ // found a number, store into Matrix
      // null-terminate the buffer so it can be passed to atof()
      buffer[bfrIdx] = '\0'; 
      
      // convert char buffer to FLOAT and store in Matrix array
      mat->arr[arrIdx++] = (FLOAT) atof(buffer);  
      
      // if array is full, double size
      if (arrIdx == matSize) { 
        doubleArraySize(mat); 
      }
      
      // only count number of columns until reach first newline
      if (countCols) numCols++;  
      
      // clear buffer, reset buffer index to 0
      memset(buffer, '\0', bfrIdx); 
      bfrIdx = 0;
    } else if(mat->mmapFileLoc[mmapIdx] == '\n') {              
      if (countCols) {
        mat->cols = numCols;  
        countCols = 0;    // disable counting columns
      }   
      numRows++;
    }
    
    /* grab a character at each loop iteration and store into buffer[] to conv to FLOAT */
    buffer[bfrIdx++] = mat->mmapFileLoc[mmapIdx];    
  }
  
  mat->rows = numRows;
}

void matrixMulOnDevice(Matrix *mat1, Matrix *mat2, Matrix *mat3){
  int size = mat1->cols * mat1->cols * sizeof(FLOAT);
  FLOAT *Md, *Nd, *Pd;
  
  // Allocate space and send data for Matrix1 to GPU
  HANDLE_ERROR( cudaMalloc(&Md, size) );
  HANDLE_ERROR( cudaMemcpy(Md, mat1->arr, size, cudaMemcpyHostToDevice) );
  
  // Allocate space and send data for Matrix2 to GPU
  HANDLE_ERROR( cudaMalloc(&Nd, size) );
  HANDLE_ERROR( cudaMemcpy(Nd, mat2->arr, size, cudaMemcpyHostToDevice) );
  
  // Allocate space for Matrix3 on GPU
  HANDLE_ERROR( cudaMalloc(&Pd, size) );
  
  // Call GPU Matrix multiply function here.
    // Not yet implemented
    
  // Copy results back to CPU
  HANDLE_ERROR( cudaMemcpy(mat3->arr, Pd, size, cudaMemcpyDeviceToHost) );
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
  unmapFile(pMat1);

  // Setup matrix2
  Matrix m2;
  Matrix *pMat2 = &m2;
  mapFileToMemory(argv[2], pMat2);
  initMatrixArray(pMat2, DEF_ROWS, DEF_COLS);
  storeMatrixToArray(pMat2);
  unmapFile(pMat2);

  // check that matrix is square
  if(pMat1->rows != pMat2->rows){
    printf("Error: matrix is not square\n");
    exit(1);
  }
 
  // Setup matrix3
  Matrix m3;
  Matrix *pMat3 = &m3;
  initMatrixArray(pMat3, pMat1->rows, pMat1->cols);
  matrixMulOnDevice(pMat1, pMat2, pMat3);
  writeOutput(pMat3);
  
  free(pMat1->arr);
  free(pMat2->arr);
  free(pMat3->arr);

  return 0;
}

