

################################ Macros #################################

SHELL = /bin/sh
CC = g++
# Enable debug options
# CFLAGS = -g -Wall -std=c++11
# Enable best optimization options
CFLAGS = -Ofast -march=native -mtune=native -std=c++11
OBJECTS = Helper.o ArffImporter.o

################################ Compile ################################

exec: ${OBJECTS} LogisticRegression.c
	$(CC) ${CFLAGS} -o $@ ${OBJECTS} LogisticRegression.c

Helper.o: Helper.c Helper.h BasicDataStructures.h
	$(CC) ${CFLAGS} -c Helper.c

ArffImporter.o: ArffImporter.cpp ArffImporter.h BasicDataStructures.h Helper.h
	$(CC) ${CFLAGS} -c ArffImporter.cpp

################################# Clean #################################

clean:
	-rm -f *.o *.h.gch exec
