#include <iostream>
#include <set>
#include <cstdlib>
#include <string>
#include <time.h>

#include "tensor.h"

#define NUM_TENSORS 40
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

  //if d=1, how many tensors will be deallocated this round
  int d_num = 3;

  //temporary tensor reference
  tensorflow::Tensor* tensor_temp;
  //copy constructor tensot reference
  tensorflow::Tensor* cp_tensor_temp;

  //give a random seed
  std::srand (time(NULL));

  for(t_cnt=1; t_cnt < num_random_t+1; ++t_cnt){
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
      if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 0){
         std::cout << "[main.cc]: Default tensor constructor is used for tensor, TID = " << t_cnt << ", BID = " << t_cnt << std::endl;
      }
      tensor_temp = new tensorflow::Tensor(t_cnt);
      tensorset.insert(tensor_temp);
    }
    else {
      //using copy tensor constructor
      //select a previous tensor index to initialize the new tensor
      r_t = std::rand()%tensorset.size()+1;//generate a random number in [1..t_cnt]
      if(std::getenv("DEBUG_FLAG") && atoi(std::getenv("DEBUG_FLAG")) == 0){
         std::cout << "[main.cc]: Copy tensor constructor is used for tensor, TID = " << t_cnt << ". Seed tensor, TID = "<< r_t << std::endl;
      }
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
    //std::cout << "[main.cc]: step1 finished: allocate/copy a tensor" << std::endl;
    //the new tensor has been added to tensorset

    // step2: decide if we want to deallocate some tensors this iteration
    d = std::rand()%2;//generate a random number in [0,1]
   
    if (d==1){
      int rnd = 0;
      while ( rnd < d_num ){
      rnd++;
        //deallocate a previous tensor this round
        if (tensorset.size()>10) {
          //we only deallocate tensor only if there are still 20 tensors alive
          d_t = std::rand()%tensorset.size()+1;//generate a random number in [1..t_cnt]
          //std::cout << "[main.cc]: before dealloc a tensor: " << tensorset.size() << std::endl;
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
        //std::cout << "[main.cc]: after dealloc a tensor: " << tensorset.size() << std::endl;
        }
      }
    }
    //std::cout << "[main.cc]: step2 finished: deallocate a tensor or not: "<< d << std::endl;

    // step3: check if the buffer size overflows? Then gc or not
    if(tensorflow::Buffer::buf_tracer.get_buffer_set_size()>tensorflow::Buffer::buf_tracer.get_thresh()){    
      std::cout << "[main.cc]: Start tracing. Current total buffer size (memory) = " << tensorflow::Buffer::buf_tracer.get_buffer_set_size() << std::endl;
      //get reference for BufTracer::tracing_set
      std::set<tensorflow::Buffer*>* tracing_set_ptr = tensorflow::Buffer::buf_tracer.get_tracing_set();
      //start tracing
      //put all reacheable object to tracing_set_ptr
      tensorflow::Tensor::tensor_tracer.start_tracing(tracing_set_ptr);
      //mark garbage and move
      tensorflow::Buffer::buf_tracer.mark_mv_garbage_set();
      //clean all garbage
      tensorflow::Buffer::buf_tracer.free_garbage_set();
      std::cout << "[main.cc]: After GC, total buffer size (memory) = " << tensorflow::Buffer::buf_tracer.get_buffer_set_size() << std::endl;
    }
    std::cout << "[main.cc] total buffer memory = " << tensorflow::Buffer::buf_tracer.get_buffer_set_size() << std::endl;
  }

  std::cout << "[main.cc] tensor set size (total tensors)="<<tensorset.size() << std::endl;
  for(tensorset_it = tensorset.begin(); tensorset_it != tensorset.end(); ++tensorset_it){
      tensor_temp = *tensorset_it;
      delete tensor_temp;
  }

  std::cout << "[main.cc]: *cleanup* Current total buffer size (memory) = " << tensorflow::Buffer::buf_tracer.get_buffer_set_size() << std::endl;
  //get reference for BufTracer::tracing_set
  std::set<tensorflow::Buffer*>* tracing_set_ptr = tensorflow::Buffer::buf_tracer.get_tracing_set();
  //start tracing
  //put all reacheable object to tracing_set_ptr
  tensorflow::Tensor::tensor_tracer.start_tracing(tracing_set_ptr);
  //mark garbage and move
  tensorflow::Buffer::buf_tracer.mark_mv_garbage_set();
  //clean all garbage
  tensorflow::Buffer::buf_tracer.free_garbage_set();
  std::cout << "[main.cc]: *cleanup* After GC, total buffer size (memory) = " << tensorflow::Buffer::buf_tracer.get_buffer_set_size() << std::endl;
  std::cout << "[main.cc] total buffer memory = " << tensorflow::Buffer::buf_tracer.get_buffer_set_size() << std::endl;
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
      tensor_temp = *tensorset_it;
      delete tensor_temp;
    }
    if (tid > NUM_TENSORS && tid < NUM_TENSORS + 11){
      tensor_temp = *tensorset_it;
      delete tensor_temp;
    }
    tid++;
  }


  //start tracing here
  std::set<tensorflow::Buffer*>* tracing_set_ptr = tensorflow::Buffer::buf_tracer.get_tracing_set();
  tensorflow::Tensor::tensor_tracer.start_tracing(tracing_set_ptr);
  tensorflow::Buffer::buf_tracer.mark_mv_garbage_set();
  tensorflow::Buffer::buf_tracer.free_garbage_set();

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

