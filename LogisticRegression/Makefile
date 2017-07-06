

################################ Macros #################################

SHELL = /bin/sh
# Enable debug options
# CFLAGS = -g -Wall -std=c++11
# Enable best optimization options
CFLAGS = -Ofast -march=native -mtune=native -std=c++11
CC = g++
OBJECTS = Helper.o ArffImporter.o

# Enable OpenAcc
PGIC = pgc++
PGICFLAGS = -Minfo=accel -acc -fast -std=c++11

################################ Compile ################################

acc_exec: ${OBJECTS} LogisticRegression.c
	$(PGIC) ${PGICFLAGS} -o $@ ${OBJECTS} LogisticRegression.c

Helper.o: Helper.c Helper.h BasicDataStructures.h
	$(PGIC) ${PGICFLAGS} -c Helper.c

ArffImporter.o: ArffImporter.cpp ArffImporter.h BasicDataStructures.h Helper.h
	$(PGIC) ${PGICFLAGS} -c ArffImporter.cpp

# exec: ${OBJECTS} LogisticRegression.c
# 	$(CC) ${CFLAGS} -o $@ ${OBJECTS} LogisticRegression.c

# Helper.o: Helper.c Helper.h BasicDataStructures.h
# 	$(CC) ${CFLAGS} -c Helper.c

# ArffImporter.o: ArffImporter.cpp ArffImporter.h BasicDataStructures.h Helper.h
# 	$(CC) ${CFLAGS} -c ArffImporter.cpp

################################# Clean #################################

clean:
	-rm -f *.o *.h.gch exec acc_exec
