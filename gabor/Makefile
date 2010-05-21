CC = g++
CUDACC = nvcc
CFLAGS = -p -g -I/usr/local/cuda/include -I./
LFLAGS = -L/usr/local/cuda/lib64 -lcudart -lcufft -lglut -lGL -lGLU -lGLEW
PROG = gabor
SRC = $(wildcard *.cpp)
OBJS = $(patsubst %.cpp,%.cpp.o,$(SRC))
CUDASRC = $(wildcard *.cu)
CUDAOBJS = $(patsubst %.cu,%.cu.o,$(CUDASRC))
LIBTARGET = libgabor.a

all : $(PROG) library

library : $(OBJS) $(CUDAOBJS)
	ar rcs $(LIBTARGET) $(OBJS) $(CUDAOBJS)

$(PROG) : $(OBJS) $(CUDAOBJS)
	$(CC) $(CFLAGS) $(LFLAGS) -o $(PROG) $(OBJS) $(CUDAOBJS) 

%.cpp.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.cu.o: %.cu
	$(CUDACC) -c -o $@ $<

clean :
	rm -f *.o
	rm $(PROG) 
	rm gmon.out
