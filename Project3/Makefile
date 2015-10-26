FLAGS=-O3 -g -debug inline-debug-info -qopenmp
SRCFILES=main.c

all:	vectorsum
		
vectorsum:  $(SRCFILES) 
	icc -D FLOAT=float $(FLAGS) -o vectorsum $^ 

vectorsum_mic:  $(SRCFILES) 
	icc -D FLOAT=float -mmic $(FLAGS) -o vectorsum $^  
	
clean: 
	rm -f *.o vectorsum

