#include "tensorflow/core/tensorgc/buftracer.h"
#include "tensorflow/core/framework/tensor.h"

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
  buffer_set.insert(newbuffer);
}

template <typename T>
void BufTracer<T>::rmfrom_buffer_set(T* oldbuffer){
  buffer_set.erase(oldbuffer);
}

template <typename T>
std::set<T*>* BufTracer<T>::get_tracing_set (){
  return &(tracing_set);
}

template <typename T>
void BufTracer<T>::mark_mv_garbage_set(){}

template <typename T>
void BufTracer<T>::free_garbage_set(){}

//template class BufTracer<Buffer>;
}// namespace tensorflow
