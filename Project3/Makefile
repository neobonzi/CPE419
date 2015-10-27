FLAGS=-O3 -qopt-report=3 -restrict -g -mkl -debug inline-debug-info -qopenmp
SRCFILES=vectorsum.c 
SRCFILES2=vectoranalyze.c

all:	vectorsum vectoranalyze
		
vectorsum:  $(SRCFILES) 
	icc $(FLAGS) -o vectorsum $^ 

# vectorsum_mic:  $(SRCFILES) 
# 	icc -mmic $(FLAGS) -o vectorsum $^  

vectoranalyze:  $(SRCFILES2) 
	icc $(FLAGS) -o vectoranalyze $^  
			
clean: 
	rm -f *.o vectorsum vectoranalyze result.out hist.a hist.b hist.c

