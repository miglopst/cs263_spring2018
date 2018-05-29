#include <iostream>
#include <cstdlib>
#include <time.h>
#include "buffer.h"

namespace tensorflow {

BufTracer<Buffer> Buffer::buf_tracer = BufTracer<Buffer>();

Buffer::Buffer(){
  std::srand (time(NULL));
  field = std::rand()%1024+1;
  buf_ = new int[field];
  buf_tracer.addto_buffer_set(this); 
  if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 5){
    std::cout << "[buffer.cc]: Buffer Constructor called: size=" << getfield() <<std::endl;
  }
}


Buffer::~Buffer(){
  if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 2){
    std::cout << "[buffer.cc]: Buffer Deconstructor called" << std::endl;
  }
  delete [] buf_;
  buf_ = NULL;
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

