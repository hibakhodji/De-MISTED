install: annot_error filter_error

clean:
	rm *.o


OBJECTS = readseq.o gcgcheck.o readxml.o writeseq.o util.o 

HEADERS = general.h clustalw.h

CC	= cc
CFLAGS  = -c -g 
LFLAGS	= -g -lm -L$(EXPAT_LIB) -lexpat
EXPAT_LIB	= /lib64
EXPAT_INC	= /lib64

annot_error : init.o $(OBJECTS) annot_error.o
	$(CC) -o $@ init.o $(OBJECTS) annot_error.o $(LFLAGS)

filter_error : init.o $(OBJECTS) filter_error.o
	$(CC) -o $@ init.o $(OBJECTS) filter_error.o $(LFLAGS)

readxml.o : readxml.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(EXPAT_INC) $*.c

.c.o :	$(HEADERS)
	$(CC) $(CFLAGS) $?

