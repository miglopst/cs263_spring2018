#include <set>
#include <iostream>
#include <cstdlib>
#include "tensorflow/core/tensorgc/roottracer.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

//T1 is Tensor
//T2 is TensorBuffer
template <typename T1, typename T2>
RootTracer<T1,T2>::RootTracer(int trace_thresh){
    this->trace_counter = 0;
    this->trace_thresh = trace_thresh;
}

template <typename T1, typename T2>
RootTracer<T1,T2>::RootTracer(const RootTracer<T1, T2> &tmp){
    typename std::set<T1*>::iterator tmp_it;
    for(tmp_it = tmp.root_set.begin(); tmp_it !=tmp.root_set.end(); ++tmp_it){
        this->addto_root_set(*tmp_it);
    }
}

template <typename T1, typename T2>
RootTracer<T1,T2>::~RootTracer(){}

template <typename T1, typename T2>
bool RootTracer<T1, T2>::find(T1* root){
    if(this->root_set.find(root) != this->root_set.end()){
        return true;
    } else {
        return false;
    }
}

template <typename T1, typename T2>
bool RootTracer<T1, T2>::compare(RootTracer<T1, T2> tmp){
    if(this->root_set.size() != tmp.root_set.size()){
        return false;
    }
    typename std::set<T1*>::iterator rootset_it;
    for(rootset_it = root_set.begin(); rootset_it!=root_set.end(); ++rootset_it){
        typename std::set<T1*>::iterator tmp_rootset_it = tmp.root_set.find(*rootset_it);
        if(tmp_rootset_it == tmp.root_set.end()){
            return false;
        }
        if((*rootset_it)->getbuf()!= (*tmp_rootset_it)->getbuf()){
            return false;
        }
    }
    return true;
}

template <typename T1, typename T2>
void RootTracer<T1,T2>::addto_root_set(T1* newtensor){
  Tensor::mtx.lock();
  root_set.insert(newtensor);
  Tensor::mtx.unlock();
  this->trace_counter += 1;
}

template <typename T1, typename T2>
void RootTracer<T1,T2>::rmfrom_root_set(T1* oldtensor){
  Tensor::mtx.lock();
  root_set.erase(oldtensor);
  Tensor::mtx.unlock();
  this->trace_counter -= 1;
}

template <typename T1, typename T2>
int RootTracer<T1,T2>::getsize_root_set(){
    return root_set.size();
}

template <typename T1, typename T2>
int RootTracer<T1,T2>::get_trace_counter(){
    return this->trace_counter;
}

template <typename T1, typename T2>
void RootTracer<T1,T2>::start_tracing(std::set<T2*>* tracing_set){
  if (root_set.size() == 0){
    LOG(ERROR) << "[Peng]tensorflow/core/tensorgc/roottracer.cc:start_tracing(),root_set is empty!";
    return;
  }

  //reset tracing_set
  //[WARNING] remove all elements from tracing_set
  Tensor::mtx.lock();
  tracing_set->clear();
  Tensor::mtx.unlock();

  std::set<Tensor*>::iterator rootset_it;
  Tensor* tensor_temp;
  for(rootset_it = root_set.begin(); rootset_it != root_set.end(); ++rootset_it){
    tensor_temp = *rootset_it;
    if (tensor_temp->getbuf() == nullptr){
      LOG(ERROR) << "[Peng]tensorflow/core/tensorgc/roottracer.cc:start_tracing(),the given tensor has a null buffer!";
      LOG(ERROR) << "[Peng]tensor addr="<<tensor_temp; 
      continue;
    }
    if ( tracing_set->find(tensor_temp->getbuf()) != tracing_set->end()){
      //this buffer is added to the tracing_set. do nothing
      LOG(ERROR) << "[Peng]tensorflow/core/tensorgc/roottracer.cc:start_tracing(),the buffer is already in the tracing_set!";
      LOG(ERROR) << "[Peng]tensor addr="<<tensor_temp;
    }
    else{
      //this buffer is not in the tracing set, and can be reached. add it to the tracing set.
      Tensor::mtx.lock();
      tracing_set->insert(tensor_temp->getbuf());
      Tensor::mtx.unlock();
      LOG(ERROR) << "[Peng]tensorflow/core/tensorgc/roottracer.cc:start_tracing(),the buffer can be traced and is added to the tracing_set!";
      LOG(ERROR) << "[Peng]tensor addr="<<tensor_temp;
      if ( tracing_set->find(tensor_temp->getbuf()->root_buffer()) != tracing_set->end() ){
        Tensor::mtx.lock();
        tracing_set->insert(tensor_temp->getbuf()->root_buffer());
        Tensor::mtx.unlock();
        LOG(ERROR) << "[Peng]tensorflow/core/tensorgc/roottracer.cc:start_tracing(),the root_buffer is added to the tracing_set!";
      }
    }
  }
  this->trace_counter = 0;
}

//initialization here is very important!
//template class RootTracer<Tensor, TensorBuffer>;

}//end tensorflow namespace
