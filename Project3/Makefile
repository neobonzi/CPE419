FLAGS=-O1 -g
SRCFILES=main.c

all:	vectorsum
		
vectorsum:  $(SRCFILES) 
	gcc -D FLOAT=float $(FLAGS) -o vectorsum $^ 
 
clean: 
	rm -f *.o vectorsum

