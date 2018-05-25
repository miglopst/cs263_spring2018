#ifndef BUFFER
#define BUFFER

#include <iostream>
#include <cstdlib>
#include "buftracer.h"


namespace tensorflow {

class Buffer{
 public:
  Buffer();
  ~Buffer();
  void setid(int i);
  int getid();
  int getfield();
  Buffer* getBref();

  static BufTracer<Buffer> buf_tracer;
  
 private:
  int id;
  int field;
};

}// namespace tensorflow
#endif
