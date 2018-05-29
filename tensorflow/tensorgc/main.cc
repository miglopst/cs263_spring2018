#include <iostream>
#include <set>
#include <cstdlib>
#include <string>
#include <time.h>

#include "tensor.h"

#define NUM_TENSORS 10
//we should not have tensorflow namespace here!!

void random_initialization_test(){
  //this test creats a number of tensors and initializes them with new allocated buffers, or share buffers with previous tensors
  //set total number of tensors to be initialized
  int num_random_t = 100;
  //tensor index [1..num_random_t]
  int t_cnt;
  //r=0: allocate new tensor; r=1: use copy constructor for tensor
  //r_t: hold a random number in [1..t_cnt]
  int r, r_t;
  //d=0: do nothing; d=1: deallocate some previous tensor this round
  //d_t: hold a random number in [1..t_cnt], for deallocation
  int d, d_t;

  //contains all references to allocated tensors
  //[WARNING] note that when a tensor is deallocated in this set, the reference is not moved out, the tensor pointer may points to an empty space!
  std::set<tensorflow::Tensor*> tensorset;
  std::set<tensorflow::Tensor*>::iterator tensorset_it;

  //this set contains all tensor index that have been deallocated
  std::set<int> tensor_dealloc;

  //temporary tensor reference
  tensorflow::Tensor* tensor_temp;
  //copy constructor tensot reference
  tensorflow::Tensor* cp_tensor_temp;

  std::srand (time(NULL));
  for(t_cnt=1; t_cnt < num_random_t+1; ++t_cnt){
    std::cout << "========rnd:" << t_cnt << std::endl;
    if (t_cnt>1){
      //for the later tensors, we randomly use default (r=0), or copy constructor (r=1).
      r = std::rand()%2;//generate a random number in [0,1]
    }
    else {
      //for the first tensor, we must use the default tensor constructor
      r = 0;
    }

    // step 1: allocate/copy a tensor
    if (r==0){
      //using default tensor constructor
      tensor_temp = new tensorflow::Tensor(t_cnt);
      tensorset.insert(tensor_temp);
    }
    else {
      //using copy tensor constructor
      //select a previous tensor index to initialize the new tensor
      r_t = std::rand()%tensorset.size()+1;//generate a random number in [1..t_cnt]
      for(tensorset_it = tensorset.begin(); tensorset_it != tensorset.end(); ++tensorset_it){
        r_t--;
        if (r_t==0){
          tensor_temp = *tensorset_it;
          cp_tensor_temp = new tensorflow::Tensor(t_cnt,tensor_temp);
          tensorset.insert(cp_tensor_temp);
          break;
        }
      }
    }
    std::cout << "step1 finished: allocate/copy a tensor" << std::endl;
    //the new tensor has been added to tensorset

    // step2: decide if we want to deallocate some tensors this iteration
    d = std::rand()%2;//generate a random number in [0,1]
    if (d==1){
      //deallocate a previous tensor this round
      //give a random seed
      if (tensorset.size()>20) {
        //we only deallocate tensor only if there are still 20 tensors alive
        //std::srand (time(NULL));
        d_t = std::rand()%tensorset.size()+1;//generate a random number in [1..t_cnt]
	    //insert into deallocation set
	    tensor_dealloc.insert(d_t);
        std::cout << "[main.cc] before dealloc a tensor: " << tensorset.size() << std::endl;
    	for(tensorset_it = tensorset.begin(); tensorset_it != tensorset.end(); ++tensorset_it){
            d_t--;
            if (d_t==0){
                tensor_temp = *tensorset_it;
                tensorset.erase(tensor_temp);
                std::cout << "[main.cc]: deallocate tensor, tid = " << tensor_temp->getid() << std::endl;
                delete tensor_temp;
                tensor_temp = NULL;
                break;
            }
        }
        std::cout << "[main.cc] after dealloc a tensor: " << tensorset.size() << std::endl;
      }
    }
    std::cout << "step2 finished: deallocate a tensor or not: "<< d << std::endl;

    // step3: check if the buffer size overflows? Then gc or not
    std::cout << tensorflow::Buffer::buf_tracer.get_tracing_set_size() << std::endl;
    if(tensorflow::Buffer::buf_tracer.get_tracing_set_size()>tensorflow::Buffer::buf_tracer.get_thresh()){    
      //get reference for BufTracer::tracing_set
      std::set<tensorflow::Buffer*>* tracing_set_ptr = tensorflow::Buffer::buf_tracer.get_tracing_set();
      //start tracing
      //put all reacheable object to tracing_set_ptr
      tensorflow::Tensor::tensor_tracer.start_tracing(tracing_set_ptr);
      //mark garbage and move
      tensorflow::Buffer::buf_tracer.mark_mv_garbage_set();
      //clean all garbage
      tensorflow::Buffer::buf_tracer.free_garbage_set();
    }
    std::cout << "step3 finished: garbage collect or not" << std::endl;
  }

  std::cout << "tensor set size="<<tensorset.size() << std::endl;
  for(tensorset_it = tensorset.begin(); tensorset_it != tensorset.end(); ++tensorset_it){
      tensor_temp = *tensorset_it;
      delete tensor_temp;
  }
  //get reference for BufTracer::tracing_set
  std::set<tensorflow::Buffer*>* tracing_set_ptr = tensorflow::Buffer::buf_tracer.get_tracing_set();
  //start tracing
  //put all reacheable object to tracing_set_ptr
  tensorflow::Tensor::tensor_tracer.start_tracing(tracing_set_ptr);
  //mark garbage and move
  tensorflow::Buffer::buf_tracer.mark_mv_garbage_set();
  //clean all garbage
  tensorflow::Buffer::buf_tracer.free_garbage_set();

}



void linear_initialization_test(){
  int tid;
  std::set<tensorflow::Tensor*> tensorset;
  std::set<tensorflow::Tensor*>::iterator tensorset_it;
  tensorflow::Tensor* tensor_temp;
  tensorflow::Tensor* cp_tensor_temp;

  //allocate NUM_TENSORS tensors linearly
  for(tid=0; tid<NUM_TENSORS; tid++){
    tensor_temp = new tensorflow::Tensor(tid);
    tensorset.insert(tensor_temp);
  }

  //allocate NUM_TENSORS copy-allocated tensors linearly
  tensorset_it = tensorset.begin();
  for(tid=0; tid<NUM_TENSORS; tid++){
    tensor_temp = *tensorset_it;
    cp_tensor_temp = new tensorflow::Tensor(tid+NUM_TENSORS, tensor_temp);
    tensorset.insert(cp_tensor_temp);
    tensorset_it++;
  }

  //deallocate the first 10 tensors
  //deallocate the first 10 copy-allocated tensors
  tid = 1;
  for (tensorset_it = tensorset.begin(); tensorset_it != tensorset.end(); ++tensorset_it){
    if (tid > 0 && tid < 11){
    }
    if (tid > NUM_TENSORS && tid < NUM_TENSORS + 11){
    }
  }


  //start tracing here
  std::set<tensorflow::Buffer*>* tracing_set_ptr = tensorflow::Buffer::buf_tracer.get_tracing_set();
  tensorflow::Tensor::tensor_tracer.start_tracing(tracing_set_ptr);

  //tensor deallocation test
  for(tensorset_it = tensorset.begin(); tensorset_it != tensorset.end(); ++tensorset_it){
    tensor_temp = *tensorset_it;
    //if (atoi(std::getenv("DEBUG_FLAG")) == 2)
    //  std::cout << "tid=" << tensor_temp->getid() << std::endl;
    delete tensor_temp;
  }
  delete cp_tensor_temp;
}


int main(){

  //check debug flag
  if(std::getenv("DEBUG_FLAG") != NULL){
    std::cout << "debug-"<< atoi(std::getenv("DEBUG_FLAG")) <<" enabled" << std::endl;
  }
  else{
    std::cout << "debug disabled, please set DEBUG_FLAG" << std::endl;
    std::cout << "$ export DEBUG_FLAG = <DEBUG_LVL>" << std::endl;
    exit(0);
  }

  //use linear initialization here
  //std::cout << "===start linear initialization ===" << std::endl;
  //linear_initialization_test();
  //std::cout << "===end linear initialization ===" << std::endl;
  
  //use random initiatization here
  std::cout << "===start random initialization ===" << std::endl;
  random_initialization_test();
  std::cout << "===end random initialization ===" << std::endl;
  return 0;
}

