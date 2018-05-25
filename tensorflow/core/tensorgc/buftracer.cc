#include "tensorflow/core/tensorgc/buftracer.h"
#include "tensorflow/core/framework/tensor.h"

#include <set>
#include <iostream>
#include <cstdlib>

namespace tensorflow {

template <typename T>
BufTracer<T>::BufTracer(int tracing_thresh){
    this->tracing_set_size = 0;
    this->tracing_thresh = tracing_thresh;
}

template <typename T>
BufTracer<T>::~BufTracer(){}

template <typename T>
void BufTracer<T>::addto_buffer_set(T* newbuffer){
  buffer_set.insert(newbuffer);
  tracing_set_size += newbuffer->size();
}

template <typename T>
void BufTracer<T>::rmfrom_buffer_set(T* oldbuffer){
  buffer_set.erase(oldbuffer);
  tracing_set_size -= oldbuffer->size();
}

template <typename T>
std::set<T*>* BufTracer<T>::get_tracing_set (){
  return &(tracing_set);
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
void BufTracer<T>::mark_mv_garbage_set(){}

template <typename T>
void BufTracer<T>::free_garbage_set(){}

//template class BufTracer<Buffer>;
}// namespace tensorflow
