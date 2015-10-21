#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/mman.h>

#define NUM_BINS 40
#define MAX_VAL 10
#define MIN_VAL -10
#define DEF_SIZE 2

typedef struct{
  float *arr;
  float *hist;
  int mmapFileSize;
  char *mmapFileLoc;
  int size;
} Vector;

/**
 * Compute histogram for a given vector
 */
void computeHistogram(Vector *v)
{
    int vectorIndex;
    int spread = MAX_VAL - MIN_VAL;
    float binWidth = spread / NUM_BINS;

    // Create the bins uu
    v->hist = malloc(sizeof(float) * NUM_BINS);
    
    for(vectorIndex = 0; vectorIndex < v.size; vectorIndex++)
    {
        //Compute bin
        int binIndex = (v[vectorIndex] - MIN_VAL) / binWidth;
        v->hist[binIndex]++;
    }
}

/**
 * Add two vectors and place the result in an output vector
 */
void addVectors(Vector *v1, Vector *v2, Vector *out)
{
    int vectorIndex;
    
    for(vectorIndex = 0; vectorIndex < v1.size; vectorIndex++)
    {
        out->arr[vectorIndex] = v1->arr[vectorIndex] + v2->arr[vectorIndex];
    }
}

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
void initVectorArray(Vector *vec, int initSize) {
  vec->size = initSize;

  float *newArray = (float *) malloc(sizeof(float) * vec->size);

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
  float *newArray = (float *) malloc(sizeof(float) * vec->size * 2);

  if (newArray == NULL) {
    perror("Error, couldn't allocate space for array\n");
    exit(1);
  }

  // copy old array to newArray
  newArray = (float*) memcpy(newArray, vec->arr, sizeof(float) * vec->size);

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
  char buffer[10];    // buffer to hold float up to 9 digits long

  for(mmapIdx = 0; mmapIdx <= vec->mmapFileSize; mmapIdx++) {
    if(vec->mmapFileLoc[mmapIdx] == ' '){ // found a number, store into Vector
      // null-terminate the buffer so it can be passed to atof()
      buffer[bfrIdx] = '\0';

      // if array is full, double size
      if (arrIdx == vec->size) {
        doubleArraySize(vec);
      }

      // convert char buffer to float and store in Vector array
      vec->arr[arrIdx++] = (float) atof(buffer);

      // clear buffer, reset buffer index to 0
      memset(buffer, '\0', bfrIdx);
      bfrIdx = 0;
    } else if(vec->mmapFileLoc[mmapIdx] == '\n') {
      // done with file IO
      return;
    }

    /* grab a character at each loop iteration and store into buffer[] to 
    conv to float */
    buffer[bfrIdx++] = vec->mmapFileLoc[mmapIdx];
  }
}

void errorCheckMatrices(Vector *vec1, Vector *vec2){
  if (vec1->size != vec2->size){
    fprintf(stderr,"Error: vectors are not compatible size\n");
    fprintf("vec1 size %d, vec2 size %d", vec1->size, vec2->size);
    exit(1);
  }
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
  
  Vector v1;
  Vector *pVec1 = &v1;
  mapFileToMemory(argv[1], pVec1);
  initVectorArray(pVec1, DEF_SIZE);
  storeVectorToArray(pVec1);
  unmapFile(pVec1)
  
  Vector v2;
  Vector *pVec2 = &v2;
  mapFileToMemory(argv[2], pVec2);
  initVectorArray(pVec2, DEF_SIZE);
  storeVectorToArray(pVec2);
  unmapFile(pVec2)
  
  errorCheckVectors(pVec1, pVec2);
 
  Vector v3;
  Vector *pVec3 = &v3;
  int max = pVec1->size > pVec2->size ? pVec1->size : pVec2->size;
  initVectorArray(pVec3, max);
  storeVectorToArray(pVec2);
  unmapFile(pVec2)

  // Free allocated memory
  free(pVec1->arr);
  free(pVec2->arr);
  free(pVec3->arr);
 
  free(pVec1->hist);
  free(pVec2->hist);
  free(pVec3->hist);


  return 0;
}
