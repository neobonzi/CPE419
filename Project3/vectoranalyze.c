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
#include <sys/time.h>

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
  double time_min;
  double time_max;
  double time_mean;
  double time_dev;
  double time_median;
  double time_sort;
} Vector;

/**
* This function will take a filename and map its contents into memory for 
* faster access.
*/
void mapFileToMemory(char* fName, Vector *vec){
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
}

/**
* This function will un-map a file that was previously mapped into memory.
*/
void unmapFile(Vector *vec){
  if (munmap(vec->mmapFileLoc, vec->mmapFileSize) == -1) {
  	perror("Error un-mapping the file");
  	exit(1);
  }
}

/**
* Write results to a file named "results.out"
* Data in mat is stored row-major order.
*/
void writeOutput(Vector *vec){
  FILE* ofp;
  ofp = fopen("analyze.out", "w");
  if(ofp == NULL) {
    perror("Could not open result.out to write results");
    exit(1);
  }

  // Minimum value
  fprintf(ofp, "Minimum value: %.2f\n", vec->min_value);
  
  // Maximum value
  fprintf(ofp, "Maximum value: %.2f\n", vec->max_value);
  
  // Mean
  fprintf(ofp, "Mean: %.2f\n", vec->mean);
  
  // Standard Deviation
  fprintf(ofp, "Standard Deviation: %.2f\n", vec->std_dev);
  
  // Median
  fprintf(ofp, "Median: %.2f\n", vec->median);
  
  // output a sorted (ascending order) copy of the array.
  fprintf(ofp, "Array: ");
  int i;
  for(i = 0; i < vec->size; i++){
    fprintf(ofp, "%.2f ", vec->arr[i]);
  }
  
  fprintf(ofp, "\n");

  // close output file pointer
  fclose(ofp);   // this line is causing a segfault, not sure why
}

/**
* Print the contents of matrix to stdout for debugging
*/
void printVector(Vector *vec) {
  int i;
  for(i = 0; i < vec->size; i++){
    printf("%.2f ", vec->arr[i]);
  }
}

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
*
*/
void storeTime(double *time, struct timeval t1, struct timeval t2){
  // compute and print the elapsed time in millisec
  double elapsed_time = 0.0;
  // elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
  // elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
  
  elapsed_time = ((double) (t2.tv_usec - t1.tv_usec) / 1000000) + 
                 (double) (t2.tv_sec - t1.tv_sec);
  *time = elapsed_time;
}

/**
*
*/
void findMedian(Vector *vec) {
  struct timeval t1, t2;
  gettimeofday(&t1, NULL);  
  int size = vec->size;
  // find the median value of vec->arr
  if (size % 2 == 1) {
    // odd sized array 
    int idx = (size - 1) / 2;
    vec->median = vec->arr[idx];
  } else {
    int idx1 = size / 2;
    int idx2 = idx1 - 1;
    vec->median = (vec->arr[idx1] + vec->arr[idx2]) / 2;
  }
  gettimeofday(&t2, NULL);
  storeTime(&(vec->time_median), t1, t2);  
}

/**
*
*/
void calcStandardDeviation(Vector *vec) {
  struct timeval t1, t2;
  gettimeofday(&t1, NULL);
  // create temp array to hold interim values
  double sum = 0.0;

  // calculate the sum of all (arr[i] - mean) ^2
  int i;
  #pragma omp parallel for
  #pragma simd
  for (i = 0; i < vec->size; i++) {
    sum += pow(vec->arr[i] - vec->mean, 2);
  }
  
  // take square root of ( sum / size - 1)
  double frac = sum / (vec->size - 1);
  vec->std_dev = sqrt(frac);
  gettimeofday(&t2, NULL);
  storeTime(&(vec->time_dev), t1, t2);
}

/**
*
*/
void findMin(Vector *vec){
  struct timeval t1, t2;
  gettimeofday(&t1, NULL);
  VSLSSTaskPtr task;
  MKL_INT num_tasks = 1;
  MKL_INT obs_size = vec->size;
  MKL_INT xstorage = VSL_SS_MATRIX_STORAGE_ROWS;
  int status;
  status = vsldSSNewTask(&task, &num_tasks, &obs_size, &xstorage, vec->arr, 0, 0);
  status = vsldSSEditTask(task, VSL_SS_ED_MIN, &(vec->min_value));
  status = vsldSSCompute(task, VSL_SS_MIN, VSL_SS_METHOD_1PASS);
  status = vslSSDeleteTask(&task);
  gettimeofday(&t2, NULL);
  storeTime(&(vec->time_min), t1, t2);
}

/**
*
*/
void findMax(Vector *vec){
  struct timeval t1, t2;
  gettimeofday(&t1, NULL);  
  VSLSSTaskPtr task;
  MKL_INT num_tasks = 1;
  MKL_INT obs_size = vec->size;
  MKL_INT xstorage = VSL_SS_MATRIX_STORAGE_ROWS;
  int status;
  status = vsldSSNewTask(&task, &num_tasks, &obs_size, &xstorage, vec->arr, 0, 0);
  status = vsldSSEditTask(task, VSL_SS_ED_MAX, &(vec->max_value));
  status = vsldSSCompute(task, VSL_SS_MAX, VSL_SS_METHOD_1PASS);
  status = vslSSDeleteTask(&task);
  gettimeofday(&t2, NULL);
  storeTime(&(vec->time_max), t1, t2);  
}

/**
*
*/
void findMean(Vector *vec){
  struct timeval t1, t2;
  gettimeofday(&t1, NULL);    
  VSLSSTaskPtr task;
  MKL_INT num_tasks = 1;
  MKL_INT obs_size = vec->size;
  MKL_INT xstorage = VSL_SS_MATRIX_STORAGE_ROWS;
  int status;
  status = vsldSSNewTask(&task, &num_tasks, &obs_size, &xstorage, vec->arr, 0, 0);
  status = vsldSSEditTask(task, VSL_SS_ED_MEAN, &(vec->mean));
  status = vsldSSCompute(task, VSL_SS_MEAN, VSL_SS_METHOD_1PASS);
  status = vslSSDeleteTask(&task);
  gettimeofday(&t2, NULL);
  storeTime(&(vec->time_mean), t1, t2);   
}

/**
* Compare function for qsort. 
* Return val < 0, num1 goes before num2
* Return val = 0, no change in order
* Return val < 0, num2 goes before num1
*/
int compare(const void *num1, const void *num2){
  double fnum1 = *(const double*) num1;
  double fnum2 = *(const double*) num2;
  return (fnum1 > fnum2) - (fnum1 < fnum2);
}

/**
*
*/
void sortAscending(Vector *vec) {
  struct timeval t1, t2;
  gettimeofday(&t1, NULL);  
  qsort(vec->arr, vec->size, sizeof(double), compare);  
  gettimeofday(&t2, NULL);
  storeTime(&(vec->time_sort), t1, t2);   
}

/**
*
*/
int main( int argc, char **argv ) {
  char *file_input = "input/result1.in";
  
  // initialize vector 1 and histogram vec1
  Vector v1;
  Vector *pVec1 = &v1;
  mapFileToMemory(file_input, pVec1);
  initVectorArray(pVec1, DEF_SIZE);
  storeVectorToArray(pVec1);
  unmapFile(pVec1);

  // find the min
  findMin(pVec1);
  
  // find the max
  findMax(pVec1);
  
  // find the mean
  findMean(pVec1);
  
  // calc standard deviation
  calcStandardDeviation(pVec1);
  
  // sort array ascending order
  sortAscending(pVec1);

  // find the median after sorting the array
  findMedian(pVec1);
  
  // write output
  writeOutput(pVec1);
  
  // free allocated vector arrays
  free(pVec1->arr);

  return 0;
}
