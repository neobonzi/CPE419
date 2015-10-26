#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include <math.h>

// #define NUM_BINS 40
// #define NUM_BINS_SUM 80
// #define MAX_VAL 10
// #define MIN_VAL -10
// #define MAX_VAL_SUM 20
// #define MIN_VAL_SUM -20
#define DEF_SIZE 2

typedef struct{
  FLOAT *arr;
  // int *hist;
  int mmapFileSize;
  char *mmapFileLoc;
  int size;
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
* Write results to a file named "results.out"
* Data in mat is stored row-major order.
*/
void writeOutput(Vector *vec){
  FILE* ofp;
  ofp = fopen("result.out", "w");
  if(ofp == NULL) {
    perror("Could not open result.out to write results");
    exit(1);
  }

  int i;
  for(i = 0; i < vec->size; i++){
    fprintf(ofp, "%.2f ", vec->arr[i]);
  }

  // close output file pointer
  //fclose(ofp);   // this line is causing a segfault, not sure why
}

/**
* Print the contents of matrix to stdout for debugging
*/
// void printVector(Vector *vec) {
//   int i;
//   for(i = 0; i < vec->size; i++){
//     printf("%.2f ", vec->arr[i]);
//   }
// }

/**
* Write results to a file named "results.out"
* Data in mat is stored row-major order.
*/
// void writeHistOutput(Vector *vec, char *fileName){
//   FILE* ofp;
//   ofp = fopen(fileName, "w");
//   if(ofp == NULL) {
//     perror("Could not open result.out to write results");
//     exit(1);
//   }
// 
//   int i;
//   for(i = 0; i < NUM_BINS; i++){
//     fprintf(ofp, "%d, %d", i, vec->hist[i]);
//     if (i < NUM_BINS - 1) {
//       fprintf(ofp, "\n");
//     }
//   }
// 
//   // close output file pointer
//   //fclose(ofp);  // this line is causing a segfault, not sure why
// }

/**
* Initialize an array in memory to hold vecrix data.
*/
void initVectorArray(Vector *vec, int initSize) {
  vec->size = initSize;

  FLOAT *newArray = (FLOAT *) malloc(sizeof(FLOAT) * vec->size);

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
* This function reads in the characters and converts them to FLOATs or doubles
* before storing in the array.
*/
void storeVectorToArray(Vector *vec){
  int mmapIdx = 0, bfrIdx = 0, arrIdx = 0, localSize = 0;
  char buffer[100];    // buffer to hold FLOAT up to 99 digits long

  for(mmapIdx = 0; mmapIdx < vec->mmapFileSize; mmapIdx++) {
    if(vec->mmapFileLoc[mmapIdx] == ' '){ // found a number, store into Vector
      // null-terminate the buffer so it can be passed to atof()
      buffer[bfrIdx] = '\0';

      // if array is full, double size
      if (arrIdx == vec->size) {
        doubleArraySize(vec);
      }

      // convert char buffer to FLOAT and store in Vector array
      vec->arr[arrIdx++] = (FLOAT) atof(buffer);
      
      // increment size variable
      localSize++;

      // clear buffer, reset buffer index to 0
      memset(buffer, '\0', bfrIdx);
      bfrIdx = 0;
    } 

    /* grab a character at each loop iteration and store into buffer[] to 
    conv to FLOAT */
    buffer[bfrIdx++] = vec->mmapFileLoc[mmapIdx];
  }
  
  vec->size = localSize;
}

void errorCheckVectors(Vector *vec1, Vector *vec2){
  if (vec1->size != vec2->size){
    fprintf(stderr, "Error: vectors are not compatible size\n");
    fprintf(stderr, "vec1 size %d, vec2 size %d\n", vec1->size, vec2->size);
    exit(1);
  }
}

int main( int argc, char **argv ) {
  // exit if not enough arguments
  if (argc != 2) {
    fprintf(stderr, "Error: incorrect amount of arguments");
    exit(1);
  }
  
  // enable mic if possible
 // mkl_mic_enable();
  
  // initialize vector 1 and histogram vec1
  Vector v1;
  Vector *pVec1 = &v1;
  mapFileToMemory(argv[1], pVec1);
  initVectorArray(pVec1, DEF_SIZE);
  storeVectorToArray(pVec1);
  unmapFile(pVec1);
  // computeHistogram(pVec1, MAX_VAL, MIN_VAL);

  // compute vectorsum and histrogram vec3
  Vector v2;
  Vector *pVec2 = &v2;
  initVectorArray(pVec2, pVec1->size);    // vec sizes must be same at this pt.
  
  // writeOutput(pVec3);
    
  // write histogram output
  // writeHistOutput(pVec1, "hist.a");
  // writeHistOutput(pVec2, "hist.b");
  // writeHistOutput(pVec3, "hist.c");

  // free allocated vector arrays
  free(pVec1->arr);
  free(pVec2->arr);
  // free(pVec3->arr);    // causing segfault for some reason
 
  // free allocated histogram arrays
  // free(pVec1->hist);
  // free(pVec2->hist);
  // free(pVec3->hist);   // causing segfault for some reason

  return 0;
}
