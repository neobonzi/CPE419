#include <stdio.h>

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

__global__ int multem( int a, int b ) {
  return a * b;
}

__global__ void mult( int a, int b, int *c ) {
  *c = multem( a, b );
}

int main( void ) {
  int c;
  int *dev_c;
  HANDLE_ERROR( cudaMalloc( (void**)&dev_c, sizeof(int) ) );

  mult<<<1,1>>>( 2, 7, dev_c );

  HANDLE_ERROR( cudaMemcpy( &c, dev_c, sizeof(int),
        cudaMemcpyDeviceToHost ) );
  printf( "2 * 7 = %d\n", c );
  HANDLE_ERROR( cudaFree( dev_c ) );

  return 0;
}

