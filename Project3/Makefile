FLAGS=-g -L/lib/i386-linux-gnu -lm -pthread
SRCFILES=main.c

all:	vectorsum
		
vectorsum:  $(SRCFILES) 
	gcc -D FLOAT=float $(FLAGS) -o vectorsum $^ 
 
clean: 
	rm -f *.o vectorsum

