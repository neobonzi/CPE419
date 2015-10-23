FLAGS=-O3 -g -mmic -debug inline-debug-info
SRCFILES=main.c

all:	vectorsum
		
vectorsum:  $(SRCFILES) 
	icc -D FLOAT=float $(FLAGS) -o vectorsum $^ 
 
clean: 
	rm -f *.o vectorsum

