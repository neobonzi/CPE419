#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/mman.h>

#define DEF_ROWS 2
#define DEF_COLS 2

typedef struct{
  float *arr;
} Vector;

/**
* This function will take a filename and map its contents into memory for 
* faster access.
*/
int mapFileToMemory(char* fName, Vector *vec){
  int fd, size;
  char *map;
  struct stat st;

  stat(fName, &st);
  size = st.st_size;      // use stat to get file size
  vec->mmapFileSize = size;

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

  vec->mmapFileLoc = map;

  close(fd);
  return  0;
}

/**
* This function will un-map a file that was previously mapped into memory.
*/
int unmapFile(Vector *vec){
  if (munmap(vec->mmapFileLoc, vec->mmapFileSize) == -1) {
  	perror("Error un-mapping the file");
  	exit(1);
  }

  return 0;
}

/**
* Initialize an array in memory to hold vecrix data.
*/
void initVectorArray(Vector *vec, int initRows, int initCols) {
  vec->rows = initRows;
  vec->cols = initCols;
  vec->size = vec->rows * vec->cols;

  FLOAT *newArray = (FLOAT *) malloc(sizeof(FLOAT) * vec->rows * vec->cols);

  if (newArray == NULL) {
    perror("Error, couldn't allocate space for array");
    exit(1);
  }

  // set pointer to new array in memory
  vec->arr = newArray;
}

/**
* If array holding Vector data is not big enough create a new one twice as big.
* Copy old array data to new array, and free old array from memory.
*/
void doubleArraySize(Vector *vec) {
  // malloc new array, double the size of previous array
  FLOAT *newArray = (FLOAT *) malloc(sizeof(FLOAT) * vec->size * 2);

  if (newArray == NULL) {
    perror("Error, couldn't allocate space for array\n");
    exit(1);
  }

  // copy old array to newArray
  newArray = (FLOAT*) memcpy(newArray, vec->arr, sizeof(FLOAT) * vec->size);

  // update size of array
  vec->size *= 2;

  // free the old array
  free(vec->arr);

  // set pointer to new array in memory
  vec->arr = newArray;
}

/**
* Read values from memory mapped location into an array for processing.
* This function reads in the characters and converts them to floats or doubles
* before storing in the array.
*/
void storeVectorToArray(Vector *vec){
  int mmapIdx = 0, bfrIdx = 0, arrIdx = 0, numRows = 0, numCols = 0, 
      countCols = 1;
  char buffer[101];    // buffer to hold FLOAT up to 100 digits long

  for(mmapIdx = 0; mmapIdx <= vec->mmapFileSize; mmapIdx++) {
    if(vec->mmapFileLoc[mmapIdx] == ' '){ // found a number, store into Vector
      // null-terminate the buffer so it can be passed to atof()
      buffer[bfrIdx] = '\0';

      // if array is full, double size
      if (arrIdx == vec->size) {
        doubleArraySize(vec);
      }

      // convert char buffer to FLOAT and store in Vector array
      vec->arr[arrIdx++] = (FLOAT) atof(buffer);

      // only count number of columns until reach first newline
      if (countCols) numCols++;

      // clear buffer, reset buffer index to 0
      memset(buffer, '\0', bfrIdx);
      bfrIdx = 0;
    } else if(vec->mmapFileLoc[mmapIdx] == '\n') {
      if (countCols) {
        vec->cols = numCols;
        countCols = 0;    // disable counting columns
      }
      numRows++;
    }

    /* grab a character at each loop iteration and store into buffer[] to 
    conv to FLOAT */
    buffer[bfrIdx++] = vec->mmapFileLoc[mmapIdx];
  }

  vec->rows = numRows;
}



/**
 * This function will multiply using the cuBLAS library.
 */

int main( int argc, char **argv ) {
  // exit if not enough arguments
  if (argc != 3) {
    printf("Error: incorrect amount of arguments");
    exit(1);
  }
  
    
  // Free allocated memory

  return 0;
}
