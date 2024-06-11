install: convseq

clean:
	rm *.o


OBJECTS = readseq.o gcgcheck.o writeseq.o util.o init.o

HEADERS = general.h clustalw.h

CC	= cc
CFLAGS  = -c -g 
LFLAGS	= -g -lm 

convseq : init.o $(OBJECTS) convseq.o
	$(CC) -o $@ $(OBJECTS) convseq.o $(LFLAGS)

.c.o :	$(HEADERS)
	$(CC) $(CFLAGS) $?

