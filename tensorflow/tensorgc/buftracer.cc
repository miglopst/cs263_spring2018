#include "buftracer.h"
#include "buffer.h"
#include <set>
#include <iostream>
#include <cstdlib>

namespace tensorflow {

template <typename T>
BufTracer<T>::BufTracer(){}

template <typename T>
BufTracer<T>::~BufTracer(){}

template <typename T>
void BufTracer<T>::addto_buffer_set(T* newbuffer){
  this->buffer_set.insert(newbuffer);
  if(atoi(std::getenv("DEBUG_FLAG")) == 4){
    std::cout << "[buftracer.cc]: A new buffer is added to the buffer set." << std::endl;
  }
}


template <typename T>
std::set<T*>* BufTracer<T>::get_tracing_set (){
  return &(this->tracing_set);
}

template <typename T>
void BufTracer<T>::mark_mv_garbage_set(){}

template <typename T>
void BufTracer<T>::free_garbage_set(){}

template class BufTracer<Buffer>;
}// namespace tensorflow
