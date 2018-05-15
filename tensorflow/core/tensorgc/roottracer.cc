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
RootTracer<T1,T2>::~RootTracer(){}

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
