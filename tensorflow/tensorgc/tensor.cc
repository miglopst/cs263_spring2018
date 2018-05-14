#include <iostream>
#include <cstdlib>
#include "tensor.h"
#include "roottracer.h"
//DEBUG_FLAG:1


namespace tensorflow {

//static class member initialization
RootTracer<Tensor, Buffer> Tensor::tensor_tracer = RootTracer<Tensor, Buffer>();

Tensor::Tensor(int tid){
  this->buf_ = new Buffer;
  this->buf_->setid(tid);
  this->id = tid;
  tensor_tracer.addto_root_set(this);
  if(atoi(std::getenv("DEBUG_FLAG")) == 1){
    std::cout << "[tensor.cc]: Tensor Constructor called, TID = " << tid << std::endl;
    std::cout << "[tensor.cc]: Tensor Buf ID = " << this->buf_->getid() << std::endl;
  }
}

Tensor::Tensor(int tid, Tensor* tensor_ptr){
  this->buf_ = tensor_ptr->getbuf();
  this->id = tid;
  tensor_tracer.addto_root_set(this);
  if(atoi(std::getenv("DEBUG_FLAG")) == 1){
    std::cout << "[tensor.cc]: Copy Tensor Constructor called, TID = " << tid << std::endl;
    std::cout << "[tensor.cc]: Tensor Buf ID = " << this->buf_->getid() << std::endl;
  }
}

Tensor::~Tensor(){ 
  tensor_tracer.rmfrom_root_set(this);
  if(atoi(std::getenv("DEBUG_FLAG")) == 1)
    std::cout << "[tensor.cc]: Tensor Deconstructor called" << std::endl;
}

int Tensor::getid(){
  return this->id;
}

Buffer* Tensor::getbuf(){
  return this->buf_;
}

Tensor* Tensor::getTref(){
  return this;
}

}// namespace tensorflow

