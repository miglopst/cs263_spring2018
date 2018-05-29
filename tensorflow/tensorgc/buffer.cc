#include <iostream>
#include <cstdlib>
#include <time.h>
#include <math.h>
#include "buffer.h"

namespace tensorflow {

BufTracer<Buffer> Buffer::buf_tracer = BufTracer<Buffer>(1024);

Buffer::Buffer(){
  field = pow(2.0, std::rand()%10);
  buf_ = new int[field];
  buf_tracer.addto_buffer_set(this); 
  if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 2){
    std::cout << "[buffer.cc]: Buffer Constructor called: size=" << getfield() <<std::endl;
  }
}


Buffer::~Buffer(){
  if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 2){
    std::cout << "[buffer.cc]: Buffer Deconstructor called, BID = " << this->getid() << std::endl;
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

