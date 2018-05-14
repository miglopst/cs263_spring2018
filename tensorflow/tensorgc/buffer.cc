#include <iostream>
#include <cstdlib>
#include "buffer.h"

namespace tensorflow {

BufTracer<Buffer> Buffer::buf_tracer = BufTracer<Buffer>();

Buffer::Buffer(){
  if(atoi(std::getenv("DEBUG_FLAG")) == 2){
    std::cout << "[buffer.cc]: Buffer Constructor called" << std::endl;
  }
  buf_tracer.addto_buffer_set(this); 
}


Buffer::~Buffer(){
  if(atoi(std::getenv("DEBUG_FLAG")) == 2){
    std::cout << "[buffer.cc]: Buffer Deconstructor called" << std::endl;
  }
}

void Buffer::setid(int i){
  id = i;
}

int Buffer::getid(){
  return id;
}

Buffer* Buffer::getBref(){
  return this;
}



}// namespace tensorflow

