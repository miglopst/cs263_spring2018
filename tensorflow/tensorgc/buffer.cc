#include <iostream>
#include <cstdlib>
#include "buffer.h"

namespace tensorflow {

BufTracer<Buffer> Buffer::buf_tracer = BufTracer<Buffer>();

Buffer::Buffer(){
  field = rand()%1024;
  buf_tracer.addto_buffer_set(this); 
  if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 2){
    std::cout << "[buffer.cc]: Buffer Constructor called: size=" << getfield() <<std::endl;
  }
}


Buffer::~Buffer(){
  if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 2){
    std::cout << "[buffer.cc]: Buffer Deconstructor called" << std::endl;
  }
}

void Buffer::setid(int i){
  id = i;
}

int Buffer::getid(){
  return id;
}

int Buffer::getfield(){
  return field;
}

Buffer* Buffer::getBref(){
  return this;
}



}// namespace tensorflow

