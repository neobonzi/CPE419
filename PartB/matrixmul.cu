#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/mman.h>

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
* If array holding matrix data is not big enough create a new one twice as big.
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

int main( int argc, char **argv ) {
  int c;
  int *dev_c;
  HANDLE_ERROR( cudaMalloc( (void**)&dev_c, sizeof(int) ) );

  //mult<<<1,1>>>( 2, 7, dev_c );

  HANDLE_ERROR( cudaMemcpy( &c, dev_c, sizeof(int),
        cudaMemcpyDeviceToHost ) );
  printf( "2 * 7 = %d\n", c );
  HANDLE_ERROR( cudaFree( dev_c ) );

  return 0;
}

