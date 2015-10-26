FLAGS=-O3 -g -debug inline-debug-info -qopenmp
SRCFILES=vectorsum.c 
SRCFILES2=vectoranalyze.c

all:	vectorsum vectoranalyze
		
vectorsum:  $(SRCFILES) 
	icc -D FLOAT=float $(FLAGS) -o vectorsum $^ 

vectorsum_mic:  $(SRCFILES) 
	icc -D FLOAT=float -mmic $(FLAGS) -o vectorsum $^  

vectoranalyze:  $(SRCFILES2) 
	icc -D FLOAT=float -mmic $(FLAGS) -o vectoranalyze $^  
			
clean: 
	rm -f *.o vectorsum vectoranalyze result.out hist.a hist.b hist.c

