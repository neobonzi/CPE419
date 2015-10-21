FLAGS=-O3 -g -debug inline-debug-info -qopenmp
SRCFILES=main.c
OMPFILES=openMpEnv.c

all:	single-col ompenv 
		
single:  $(SRCFILES) 
	icc -D FLOAT=float -D COL_MAJOR=0 -D $(FLAGS) -o mm $^ 

double:  $(SRCFILES) 
	icc -D FLOAT=double -D COL_MAJOR=0 -D $(FLAGS) -o mm $^ 
	
single-col:  $(SRCFILES) 
	icc -D FLOAT=float -D COL_MAJOR=1 -D $(FLAGS) -o mm $^ 

double-col:  $(SRCFILES) 
	icc -D FLOAT=double -D COL_MAJOR=1 -D $(FLAGS) -o mm $^ 	
		
ompenv:  $(OMPFILES) 
	icc $(FLAGS) -o my-omp-env $^
 
clean: 
	rm -f *.o mm_openmp my-omp-env

