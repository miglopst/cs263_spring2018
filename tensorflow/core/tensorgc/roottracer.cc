#include <set>
#include <iostream>
#include <cstdlib>
#include "tensorflow/core/tensorgc/roottracer.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

//T1 is Tensor
//T2 is TensorBuffer
template <typename T1, typename T2>
RootTracer<T1,T2>::RootTracer(){}

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
        if(!tmp.find(*rootset_it)){
            return false;
        }
    }
    return true;
}
  //

template <typename T1, typename T2>
void RootTracer<T1,T2>::addto_root_set(T1* newtensor){
  root_set.insert(newtensor);
}

template <typename T1, typename T2>
void RootTracer<T1,T2>::rmfrom_root_set(T1* oldtensor){
  root_set.erase(oldtensor);
}

template <typename T1, typename T2>
void RootTracer<T1,T2>::start_tracing(std::set<T2*>* tracing_set){
  std::set<Tensor*>::iterator rootset_it;
  Tensor* tensor_temp;
  for(rootset_it = root_set.begin(); rootset_it != root_set.end(); ++rootset_it){
    tensor_temp = *rootset_it;
    if ( tracing_set->find(tensor_temp->getbuf()) != tracing_set->end()){
      //this buffer is added to the tracing_set. do nothing      
    }
    else{
      //this buffer is not in the tracing set, and can be reached. add it to the tracing set.
      tracing_set->insert(tensor_temp->getbuf());
    }
  }
}

//initialization here is very important!
//template class RootTracer<Tensor, TensorBuffer>;

}//end tensorflow namespace
