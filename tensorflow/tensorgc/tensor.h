#ifndef TENSOR
#define TENSOR
#include <iostream>
#include "roottracer.h"
#include "buffer.h"

namespace tensorflow {

class Tensor{
 public:
  Tensor(int tid);
  Tensor(int tid, Tensor* tensor_ptr);
  ~Tensor();
  int getid();
  Buffer* getbuf();
  Tensor* getTref();

  static RootTracer<Tensor, Buffer> tensor_tracer;
  Buffer* buf_;

 private:
  int id;
};


}// namespace tensorflow
#endif
