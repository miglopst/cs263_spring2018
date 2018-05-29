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
  if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 5){
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
  typename std::set<T*>::iterator buffer_set_it;
  typename std::set<T*>::iterator garbage_set_it;

  T* temp_buf_ptr;
  for (buffer_set_it = buffer_set.begin(); buffer_set_it != buffer_set.end(); ++buffer_set_it){
    temp_buf_ptr = *buffer_set_it;
    if (tracing_set.find(temp_buf_ptr) == tracing_set.end()){
      //the given buffer is not in the tracing set!
      //move it to the garbage_set
      if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 4){
        std::cout << "[buftracer.cc]: Buffer, bid = " << temp_buf_ptr->getid() << " is moved to garbage_set." << std::endl;
      }
      garbage_set.insert(temp_buf_ptr);
    }
    else{
      //the given buffer is in the tracing set!
      if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 5){
        std::cout << "[buftracer.cc]: Buffer, bid = " << temp_buf_ptr->getid() << " can be traced, not moving to garbage_set." << std::endl;
      }
    }
  }
  //remove all garbage elements from buffer_set

  if(garbage_set.size() > 0) {
      if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 4){
          std::cout << "[buftracer.cc]: before remove, buf set size = " << buffer_set.size() << std::endl;
      }
      for (garbage_set_it = garbage_set.begin(); garbage_set_it != garbage_set.end(); ++garbage_set_it){
          temp_buf_ptr = *garbage_set_it;
          buffer_set.erase(temp_buf_ptr);
      }
      if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 4){
          std::cout << "[buftracer.cc]: after remove, buf set size = " << buffer_set.size() << std::endl;
      }
  }

  //[WARNING] remove all elements from tracing_set
  tracing_set.clear();
}

template <typename T>
void BufTracer<T>::free_garbage_set(){
    if(garbage_set.size() > 0){
        if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 4){
            std::cout << "[buftracer.cc]: garbage_set is going to be freed, size = " << garbage_set.size() << std::endl;
            std::cout << "[buftracer.cc]: before free, buf set size = " << buffer_set.size() << std::endl;
        }
        typename std::set<T*>::iterator garbage_set_it;
        for (garbage_set_it = garbage_set.begin(); garbage_set_it != garbage_set.end(); ++garbage_set_it){
            if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 4){
                std::cout << "[buftracer.cc] trying to delete garbage buffer id =" << (*garbage_set_it)->getid()<<std::endl;
            }
            delete *garbage_set_it;
            //*garbage_set_it = NULL;
        }
        garbage_set.clear();
        if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 4){
            std::cout << "[buftracer.cc]: after free, buf set size = " << buffer_set.size() << std::endl;
        }
    }
}

template class BufTracer<Buffer>;
}// namespace tensorflow
