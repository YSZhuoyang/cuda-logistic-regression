

################################ Macros #################################

SHELL = /bin/sh
CC = g++
# Enable debug options
# CFLAGS = -g -Wall -std=c++11
# Enable best optimization options
CFLAGS = -Ofast -march=native -mtune=native -std=c++11
OBJECTS = Helper.o ArffImporter.o

# Enable Nvidia gpu
NVCC = nvcc
NVCCCFLAGS = -arch=sm_50 -std=c++11 -use_fast_math# -lcublas -lcublas_device -rdc=true -lcudadevrt

################################ Compile ################################

gpu_exec: ${OBJECTS} LogisticRegression.cu
	$(NVCC) ${NVCCCFLAGS} -o $@ ${OBJECTS} LogisticRegression.cu

Helper.o: Helper.c Helper.h BasicDataStructures.h
	$(CC) ${CFLAGS} -c Helper.c

ArffImporter.o: ArffImporter.cpp ArffImporter.h BasicDataStructures.h Helper.h
	$(CC) ${CFLAGS} -c ArffImporter.cpp

# exec: ${OBJECTS} LogisticRegression.c
# 	$(CC) ${CFLAGS} -o $@ ${OBJECTS} LogisticRegression.c

# Helper.o: Helper.c Helper.h BasicDataStructures.h
# 	$(CC) ${CFLAGS} -c Helper.c

# ArffImporter.o: ArffImporter.cpp ArffImporter.h BasicDataStructures.h Helper.h
# 	$(CC) ${CFLAGS} -c ArffImporter.cpp

################################# Clean #################################

clean:
	-rm -f *.o *.h.gch *exec*
