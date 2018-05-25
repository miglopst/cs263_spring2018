#include "buftracer.h"
#include "buffer.h"
#include <set>
#include <iostream>
#include <cstdlib>

namespace tensorflow {

template <typename T>
BufTracer<T>::BufTracer(int thresh):tracing_set_size(0), tracing_thresh(thresh){}

template <typename T>
BufTracer<T>::~BufTracer(){}

template <typename T>
void BufTracer<T>::addto_buffer_set(T* newbuffer){
  this->buffer_set.insert(newbuffer);
  this->tracing_set_size += newbuffer->getfield();
  if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 4){
    std::cout << "[buftracer.cc]: A new buffer is added to the buffer set." << std::endl;
  }
}

template <typename T>
void BufTracer<T>::rmfrom_buffer_set(T* oldbuffer){
    buffer_set.erase(oldbuffer);
    tracing_set_size -= oldbuffer->getfield();
}

template <typename T>
std::set<T*>* BufTracer<T>::get_tracing_set (){
  return &(this->tracing_set);
}

template <typename T>
int BufTracer<T>::get_tracing_set_size (){
    return tracing_set_size;
}

template <typename T>
int BufTracer<T>::get_thresh(){
    return tracing_thresh;
}

template <typename T>
void BufTracer<T>::mark_mv_garbage_set(){
  std::set<T*>::iterator tracing_set_it;
  T* temp_buf_ptr;
}

template <typename T>
void BufTracer<T>::free_garbage_set(){}

template class BufTracer<Buffer>;
}// namespace tensorflow
