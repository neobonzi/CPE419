NVFLAGS=-O3 -g -ccbin g++ -m64 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52
# list .c and .cu source files here
SRCFILES=matrixmul.cu 

all:	mm_cuda	

mm_cuda: $(SRCFILES) 
	nvcc $(NVFLAGS) -o mm_cuda $^ 

clean: 
	rm -f *.o mm_cuda

