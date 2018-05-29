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
  if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 1){
    std::cout << "[tensor.cc]: Tensor Constructor called, TID = " << tid << std::endl;
    std::cout << "[tensor.cc]: (default constructor) Tensor Buf ID = " << this->buf_->getid() << std::endl;
  }
}

Tensor::Tensor(int tid, Tensor* tensor_ptr){
  this->buf_ = tensor_ptr->getbuf();
  this->id = tid;
  tensor_tracer.addto_root_set(this);
  if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 1){
    std::cout << "[tensor.cc]: Copy Tensor Constructor called, TID = " << tid <<". Seed TID = "<< tensor_ptr->getid() << std::endl;
    std::cout << "[tensor.cc]: (copy constructor) Tensor Buf ID = " << this->buf_->getid() << std::endl;
  }
}

Tensor::~Tensor(){
  if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 1){
    std::cout << "[tensor.cc]: Tensor Deconstructor called, TID = " << this->id << std::endl;
    std::cout << "[tensor.cc]: (deconstructor) Tensor Buf ID = " << this->buf_->getid() << std::endl;
  }
  this->buf_ = NULL;
  tensor_tracer.rmfrom_root_set(this);
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

