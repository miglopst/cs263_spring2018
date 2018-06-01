#include "tensorflow/core/tensorgc/buftracer.h"
#include "tensorflow/core/framework/tensor.h"

#include <set>
#include <iostream>
#include <cstdlib>
#include <assert.h>

namespace tensorflow {

template <typename T>
BufTracer<T>::BufTracer(int tracing_thresh){
  this->buffer_set_size = 0;
  this->tracing_thresh = tracing_thresh;
}

template <typename T>
BufTracer<T>::~BufTracer(){}

template <typename T>
void BufTracer<T>::addto_buffer_set(T* newbuffer){
  Tensor::mtx.lock();
  buffer_set.insert(newbuffer);
  buffer_set_size += newbuffer->size();
  Tensor::mtx.unlock();
}

template <typename T>
void BufTracer<T>::rmfrom_buffer_set(T* oldbuffer){
  Tensor::mtx.lock();
  buffer_set.erase(oldbuffer);
  buffer_set_size -= oldbuffer->size();
  Tensor::mtx.unlock();
}

template <typename T>
std::set<T*>* BufTracer<T>::get_tracing_set (){
  return &(tracing_set);
}

template <typename T>
int BufTracer<T>::get_buffer_set_size (){
    return buffer_set_size;
}

template <typename T>
int BufTracer<T>::get_thresh(){
    return tracing_thresh;
}

template <typename T>
void BufTracer<T>::mark_mv_garbage_set(){
  typename std::set<T*>::iterator buffer_set_it;
  typename std::set<T*>::iterator garbage_set_it;
  T* temp_buf_ptr;
  for (buffer_set_it = buffer_set.begin(); buffer_set_it != buffer_set.end(); ++buffer_set_it){
    temp_buf_ptr = *buffer_set_it;
    if (temp_buf_ptr == nullptr){
      continue;
    }
    if (tracing_set.find(temp_buf_ptr) == tracing_set.end()){
      //the given buffer is not in the tracing set!
      //move it to the garbage_set
      Tensor::mtx.lock();
      garbage_set.insert(temp_buf_ptr);
      Tensor::mtx.unlock();
      LOG(ERROR) << "[Peng]tensorflow/core/tensorgc.cc:mark_mv_garbage_set(),the given buffer is moved to garbage_set.";
    }
    else{
      //the given buffer is in the tracing set!
      LOG(ERROR) << "[Peng]tensorflow/core/tensorgc.cc:mark_mv_garbage_set(),the given buffer can be traced.";
    }
  }

  //assert( (tracing_set.size()+garbage_set.size()) == buffer_set.size());
  //remove all garbage elements from buffer_set
  /*
  if(garbage_set.size() > 0) {
    for (garbage_set_it = garbage_set.begin(); garbage_set_it != garbage_set.end(); ++garbage_set_it){
      temp_buf_ptr = *garbage_set_it;
      LOG(ERROR) << "[Peng]tensorflow/core/tensorgc.cc:mark_mv_garbage_set(),rmfrom buffer set, rm size="<<temp_buf_ptr->size();
      this->rmfrom_buffer_set(temp_buf_ptr);
    }
  }
  else{
    LOG(ERROR) << "[Peng]tensorflow/core/tensorgc.cc:mark_mv_garbage_set(),garbage_set is empty.";
  }
  */
  //assert(tracing_set.size() == buffer_set.size());
  //[WARNING] remove all elements from tracing_set
  Tensor::mtx.lock();
  tracing_set.clear();
  Tensor::mtx.unlock();
}

template <typename T>
void BufTracer<T>::free_garbage_set(){
  if(garbage_set.size() > 0){
    LOG(ERROR) << "[Peng]tensorflow/core/tensorgc.cc:free_garbage_set(),start freeing garbage";
    typename std::set<T*>::iterator garbage_set_it;
    for (garbage_set_it = garbage_set.begin(); garbage_set_it != garbage_set.end(); ++garbage_set_it){
      //The garbage elements have already been removed in mark_mv_garbage_set()
      //TensorBuffer::buf_tracer.rmfrom_buffer_set(*garbage_set_it);

      Tensor::mtx.lock();
      delete *garbage_set_it;
      Tensor::mtx.unlock();
    }
    Tensor::mtx.lock();
    garbage_set.clear();
    Tensor::mtx.unlock();
  }
}

//template class BufTracer<Buffer>;
}// namespace tensorflow
