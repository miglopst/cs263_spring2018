#ifndef BUFFER
#define BUFFER

#include <iostream>
#include "buftracer.h"


namespace tensorflow {

class Buffer{
 public:
  Buffer();
  ~Buffer();
  void setid(int i);
  int getid();
  Buffer* getBref();

  static BufTracer<Buffer> buf_tracer;
  
 private:
  int id;
};

}// namespace tensorflow
#endif
