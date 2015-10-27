#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include <math.h>
#include <mkl_vsl.h>
#include <mkl.h>

#define DEF_SIZE 2

typedef struct{
  double *arr;
  int mmapFileSize;
  char *mmapFileLoc;
  int size;
  double min_value;
  double max_value;
  double mean;
  double std_dev;
  double median;
  double covariance;
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

  // Minimum value
  fprintf(ofp, "Minimum value: %.2d\n", vec->min_value);
  
  // Maximum value
  fprintf(ofp, "Maximum value: %.2d\n", vec->max_value);
  
  // Mean
  fprintf(ofp, "Mean: %.2d\n", vec->mean);
  
  // Standard Deviation
  fprintf(ofp, "Standard Deviation: %.2d\n", vec->std_dev);
  
  // Median
  fprintf(ofp, "Median: %.2d\n", vec->median);
  
  // output a sorted (ascending order) copy of the array.
  fprintf(ofp, "Mean: %.2d\n", vec->arr);
  
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
* Initialize an array in memory to hold vecrix data.
*/
void initVectorArray(Vector *vec, int initSize) {
  vec->size = initSize;

  double *newArray = (double *) malloc(sizeof(double) * vec->size);

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
  double *newArray = (double *) malloc(sizeof(double) * vec->size * 2);

  if (newArray == NULL) {
    perror("Error, couldn't allocate space for array\n");
    exit(1);
  }

  // copy old array to newArray
  newArray = (double*) memcpy(newArray, vec->arr, sizeof(double) * vec->size);

  // update size of array
  vec->size *= 2;

  // free the old array
  free(vec->arr);

  // set pointer to new array in memory
  vec->arr = newArray;
}

/**
* Read values from memory mapped location into an array for processing.
* This function reads in the characters and converts them to doubles or doubles
* before storing in the array.
*/
void storeVectorToArray(Vector *vec){
  int mmapIdx = 0, bfrIdx = 0, arrIdx = 0, localSize = 0;
  char buffer[100];    // buffer to hold double up to 99 digits long

  for(mmapIdx = 0; mmapIdx < vec->mmapFileSize; mmapIdx++) {
    if(vec->mmapFileLoc[mmapIdx] == ' '){ // found a number, store into Vector
      // null-terminate the buffer so it can be passed to atof()
      buffer[bfrIdx] = '\0';

      // if array is full, double size
      if (arrIdx == vec->size) {
        doubleArraySize(vec);
      }

      // convert char buffer to double and store in Vector array
      vec->arr[arrIdx++] = (double) atof(buffer);
      
      // increment size variable
      localSize++;

      // clear buffer, reset buffer index to 0
      memset(buffer, '\0', bfrIdx);
      bfrIdx = 0;
    } 

    /* grab a character at each loop iteration and store into buffer[] to 
    conv to double */
    buffer[bfrIdx++] = vec->mmapFileLoc[mmapIdx];
  }
  
  vec->size = localSize;
}

/**
* Compare function for qsort. 
* Return val < 0, num1 goes before num2
* Return val = 0, no change in order
* Return val < 0, num2 goes before num1
*/
int compare(const void *num1, const void *num2){
  float fnum1 = *(const float*) num1;
  float fnum2 = *(const float*) num2;
  return (fnum1 > fnum2) - (fnum1 < fnum2);
}

int main( int argc, char **argv ) {
  char *file_input = "result.out";
  
  // initialize vector 1 and histogram vec1
  Vector v1;
  Vector *pVec1 = &v1;
  mapFileToMemory(file_input, pVec1);
  initVectorArray(pVec1, DEF_SIZE);
  storeVectorToArray(pVec1);
  unmapFile(pVec1);

  // initialize MKL variables
  VSLSSTaskPtr task;
  MKL_INT num_tasks = 1;
  MKL_INT obs_size = pVec1->size;
  MKL_INT xstorage = VSL_SS_MATRIX_STORAGE_ROWS;
  int status;
  double *weight = 0;
  
  // compute the following:
    // Minimum value
    // Maximum value
    // Mean
    // Standard Deviation
    // Median
    // output a sorted (ascending order) copy of the array.
    
  // Step 1. create the task
  status = vsldSSNewTask(&task, &num_tasks, &obs_size, &xstorage, pVec1->arr, 
                         weight, 0);
      
  // Step 2. edit task parameters
  status = vsldSSEditTask(task, VSL_SS_ED_MIN, &(pVec1->min_value));
  status = vsldSSEditTask(task, VSL_SS_ED_MAX, &(pVec1->max_value));
  status = vsldSSEditTask(task, VSL_SS_ED_MEAN, &(pVec1->mean));
  status = vsldSSEditTask(task, VSL_SS_ED_VARIATION, &(pVec1->covariance));
  
  // step 3. computation of serveral estimates using 1PASS method
  MKL_INT estimates = VSL_SS_ED_MIN|VSL_SS_ED_MAX|VSL_SS_ED_MEAN|VSL_SS_ED_VARIATION;
  status = vsldSSCompute(task, estimates, VSL_SS_METHOD_1PASS);
  
  // step 4. de-allocate task resources
  status = vslSSDeleteTask(&task);
  
  // compute standard deviation using variance & mean
  pVec1->std_dev = pVec1->covariance * pVec1->mean;
  
  // sort array ascending order
  qsort(pVec1->arr, pVec1->size, sizeof(float), compare);
  
  // write output
  writeOutput(pVec1);
    
  // free allocated vector arrays
  free(pVec1->arr);

  return 0;
}
