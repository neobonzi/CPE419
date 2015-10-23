FLAGS=-O3 -g -debug inline-debug-info -mmic
SRCFILES=main.c

all:	vectorsum
		
vectorsum:  $(SRCFILES) 
	icc -D FLOAT=float $(FLAGS) -o vectorsum $^ 
 
clean: 
	rm -f *.o vectorsum

