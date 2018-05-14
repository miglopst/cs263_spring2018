OBJECTS = main.o tensor.o buffer.o roottracer.o buftracer.o

tracing: $(OBJECTS)
	$(CXX) -o tracing $(OBJECTS)

main.o: main.cc tensor.h 
	$(CXX) -c main.cc

tensor.o: tensor.cc tensor.h roottracer.h buffer.h
	$(CXX) -c tensor.cc

buffer.o: buffer.cc buffer.h buftracer.h
	$(CXX) -c buffer.cc

roottracer.o: roottracer.cc roottracer.h
	$(CXX) -c roottracer.cc

buftracer.o: buftracer.cc buftracer.h
	$(CXX) -c buftracer.cc

clean:
	rm *.o
