OBJECTS = main.o tensor.o buffer.o roottracer.o buftracer.o
CXXFLAGS := -std=c++11 -g -Wall


tracing: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o tracing $(OBJECTS)

main.o: main.cc tensor.h 
	$(CXX) $(CXXFLAGS) -c main.cc

tensor.o: tensor.cc tensor.h roottracer.h buffer.h
	$(CXX) $(CXXFLAGS) -c tensor.cc

buffer.o: buffer.cc buffer.h buftracer.h
	$(CXX) $(CXXFLAGS) -c buffer.cc

roottracer.o: roottracer.cc roottracer.h
	$(CXX) $(CXXFLAGS) -c roottracer.cc

buftracer.o: buftracer.cc buftracer.h
	$(CXX) $(CXXFLAGS) -c buftracer.cc

clean:
	rm *.o
