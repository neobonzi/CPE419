#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include <mkl.h>
#include <math.h>

#define NUM_BINS 40
#define MAX_VAL 10
#define MIN_VAL -10
#define MAX_VAL_SUM 20
#define MIN_VAL_SUM -20
#define DEF_SIZE 2

typedef struct{
  float *arr;
  int *hist;
  int mmapFileSize;
  char *mmapFileLoc;
  int size;
} Vector;

/**
 * Compute histogram for a given vector
 */
void computeHistogram(Vector* restrict v, int max, int min)
{
    // fprintf(stderr, "comp hist for vector size: %d", v->size);
    int vectorIndex = 0;
    int binIndex = 0;
    int spread = max - min;
    float binWidth =  ((float) spread) / NUM_BINS;
    
    // Create the bins
    v->hist = malloc(sizeof(int) * NUM_BINS);
    
    // set all bins to zero
    memset(v->hist, 0, sizeof(int) * NUM_BINS);
    
    for(vectorIndex = 0; vectorIndex < v->size; vectorIndex++)
    {
        //Compute bin
        if(v->arr[vectorIndex] == max)
        {
            v->hist[NUM_BINS - 1]++;
        }
        else
        {
            binIndex = (v->arr[vectorIndex] - min) / binWidth;
            v->hist[binIndex]++;
        }
    }
}


/**
 * Add two vectors and place the result in an output vector
 */
void addVectors(Vector* restrict v1, Vector* restrict v2, Vector* restrict out)
{
    int vectorIndex;

    #pragma offload target(mic) in(v1:length(v1->size)) \
                                in(v2:length(v2->size)) \
                                out(out:length(v1->size)) 
    for(vectorIndex = 0; vectorIndex < v1->size; vectorIndex++)
    {
        out->arr[vectorIndex] = v1->arr[vectorIndex] + v2->arr[vectorIndex];
    }
}

/**
* This function will take a filename and map its contents into memory for 
* faster access.
*/
int mapFileToMemory(char* restrict fName, Vector* restrict vec){
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
void printVector(Vector *vec) {
  int i;
  for(i = 0; i < vec->size; i++){
    printf("%.2f ", vec->arr[i]);
  }
}

/**
* Write results to a file named "results.out"
* Data in mat is stored row-major order.
*/
void writeHistOutput(Vector *vec, char *fileName){
  FILE* ofp;
  ofp = fopen(fileName, "w");
  if(ofp == NULL) {
    perror("Could not open result.out to write results");
    exit(1);
  }

  int i;
  for(i = 0; i < NUM_BINS; i++){
    fprintf(ofp, "%d, %d", i, vec->hist[i]);
    if (i < NUM_BINS - 1) {
      fprintf(ofp, "\n");
    }
  }

  // close output file pointer
  //fclose(ofp);  // this line is causing a segfault, not sure why
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
  int mmapIdx = 0, bfrIdx = 0, arrIdx = 0, localSize = 0;
  char buffer[100];    // buffer to hold float up to 99 digits long

  for(mmapIdx = 0; mmapIdx < vec->mmapFileSize; mmapIdx++) {
    if(vec->mmapFileLoc[mmapIdx] == ' '){ // found a number, store into Vector
      // null-terminate the buffer so it can be passed to atof()
      buffer[bfrIdx] = '\0';

      // if array is full, double size
      if (arrIdx == vec->size) {
        doubleArraySize(vec);
      }

      // convert char buffer to float and store in Vector array
      vec->arr[arrIdx++] = (float) atof(buffer);
      
      // increment size variable
      localSize++;

      // clear buffer, reset buffer index to 0
      memset(buffer, '\0', bfrIdx);
      bfrIdx = 0;
    } 

    /* grab a character at each loop iteration and store into buffer[] to 
    conv to float */
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
  if (argc != 3) {
    fprintf(stderr, "Error: incorrect amount of arguments");
    exit(1);
  }
  
  // initialize vector 1 and histogram vec1
  Vector v1;
  Vector *pVec1 = &v1;
  mapFileToMemory(argv[1], pVec1);
  initVectorArray(pVec1, DEF_SIZE);
  storeVectorToArray(pVec1);
  unmapFile(pVec1);
  computeHistogram(pVec1, MAX_VAL, MIN_VAL);

  // initialize vector 2 and histogram vec2
  Vector v2;
  Vector *pVec2 = &v2;
  mapFileToMemory(argv[2], pVec2);
  initVectorArray(pVec2, DEF_SIZE);
  storeVectorToArray(pVec2);
  unmapFile(pVec2);
  computeHistogram(pVec2, MAX_VAL, MIN_VAL);

  // error check vectors
  errorCheckVectors(pVec1, pVec2);

  // compute vectorsum and histrogram vec3
  Vector v3;
  Vector *pVec3 = &v3;
  initVectorArray(pVec3, pVec1->size);    // vec sizes must be same at this pt.
  vsAdd(pVec1->size, pVec1->arr, pVec2->arr, pVec3->arr); 
  computeHistogram(pVec3, MAX_VAL_SUM, MIN_VAL_SUM);
  writeOutput(pVec3);
    
  // write histogram output
  writeHistOutput(pVec1, "hist.a");
  writeHistOutput(pVec2, "hist.b");
  writeHistOutput(pVec3, "hist.c");

  // free allocated vector arrays
  free(pVec1->arr);
  free(pVec2->arr);
  free(pVec3->arr);    // causing segfault for some reason
 
  // free allocated histogram arrays
  free(pVec1->hist);
  free(pVec2->hist);
  free(pVec3->hist);   // causing segfault for some reason

  return 0;
}
