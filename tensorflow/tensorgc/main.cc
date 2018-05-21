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
	int num_random_t = 100;
	int t_cnt;
	int r,r_t;
	std::set<tensorflow::Tensor*> tensorset;
	std::set<tensorflow::Tensor*>::iterator tensorset_it;
	tensorflow::Tensor* tensor_temp;
	tensorflow::Tensor* cp_tensor_temp;
	for(t_cnt=1; t_cnt < num_random_t+1; ++t_cnt){
		if (t_cnt>1)
			r = std::rand()%2;//generate a random number in [0,1]
		else
			r = 0;
		
		if (r==0){
			//using default tensor constructor
			tensor_temp = new tensorflow::Tensor(t_cnt);
			tensorset.insert(tensor_temp);
		}
		else{
            //give a random seed
            std::srand (time(NULL));
			//using copy tensor constructor
			r_t = std::rand()%tensorset.size()+1;//generate a random number in [1..t_cnt]
			//std::cout << "set size = " << tensorset.size()  << std::endl;
			//std::cout << r_t << std::endl;
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
	}
  
  // backup root tracer
  tensorflow::RootTracer<tensorflow::Tensor, tensorflow::Buffer> backup_tensor_tracer(tensorflow::Tensor::tensor_tracer); 
  if(backup_tensor_tracer.compare(tensorflow::Tensor::tensor_tracer)) {
      std::cout << "root set has not changed before tracing: " << tensorflow::Tensor::tensor_tracer.getsize_root_set() << std::endl;
  } else {
      std::cout << "root set has changed before tracing: " << tensorflow::Tensor::tensor_tracer.getsize_root_set() << std::endl;
  }

  //start tracing here
  std::set<tensorflow::Buffer*>* tracing_set_ptr = tensorflow::Buffer::buf_tracer.get_tracing_set();
  tensorflow::Tensor::tensor_tracer.start_tracing(tracing_set_ptr);

  // compare whether the root set has changed after tracing
  if(backup_tensor_tracer.compare(tensorflow::Tensor::tensor_tracer)) {
      std::cout << "root set has not changed after tracing: " << tensorflow::Tensor::tensor_tracer.getsize_root_set() << std::endl;
  } else {
      std::cout << "root set has changed after tracing: " << tensorflow::Tensor::tensor_tracer.getsize_root_set() << std::endl;
  }

  //using default tensor constructor
  tensor_temp = new tensorflow::Tensor(t_cnt);
  tensorset.insert(tensor_temp);

  // compare whether the root set has changed after tracing
  if(backup_tensor_tracer.compare(tensorflow::Tensor::tensor_tracer)) {
      std::cout << "root set has not changed after tracing: " << tensorflow::Tensor::tensor_tracer.getsize_root_set() << std::endl;
  } else {
      std::cout << "root set has changed after tracing: " << tensorflow::Tensor::tensor_tracer.getsize_root_set() << std::endl;
  }

}



void linear_initialization_test(){
  int tid;
  std::set<tensorflow::Tensor*> tensorset;
  std::set<tensorflow::Tensor*>::iterator tensorset_it;
  tensorflow::Tensor* tensor_temp;
  tensorflow::Tensor* cp_tensor_temp;

  for(tid=0; tid<NUM_TENSORS; tid++){
    tensor_temp = new tensorflow::Tensor(tid);
    tensorset.insert(tensor_temp);
  }
  cp_tensor_temp = new tensorflow::Tensor(NUM_TENSORS, tensor_temp);

  // backup root tracer
  tensorflow::RootTracer<tensorflow::Tensor, tensorflow::Buffer> backup_tensor_tracer(tensorflow::Tensor::tensor_tracer); 
  if(backup_tensor_tracer.compare(tensorflow::Tensor::tensor_tracer)) {
      std::cout << "root set has not changed before tracing: " << tensorflow::Tensor::tensor_tracer.getsize_root_set() << std::endl;
  } else {
      std::cout << "root set has changed before tracing: " << tensorflow::Tensor::tensor_tracer.getsize_root_set() << std::endl;
  }

  //start tracing here
  std::set<tensorflow::Buffer*>* tracing_set_ptr = tensorflow::Buffer::buf_tracer.get_tracing_set();
  tensorflow::Tensor::tensor_tracer.start_tracing(tracing_set_ptr);

  // compare whether the root set has changed after tracing
  if(backup_tensor_tracer.compare(tensorflow::Tensor::tensor_tracer)) {
      std::cout << "root set has not changed after tracing: " << tensorflow::Tensor::tensor_tracer.getsize_root_set() << std::endl;
  } else {
      std::cout << "root set has changed after tracing: " << tensorflow::Tensor::tensor_tracer.getsize_root_set() << std::endl;
  }

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
	//if(atoi(std::getenv("DEBUG_FLAG")) != 0){
	if(std::getenv("DEBUG_FLAG") != NULL){
		std::cout << "debug-"<< atoi(std::getenv("DEBUG_FLAG")) <<" enabled" << std::endl;
	}
	else{
		std::cout << "debug disabled" << std::endl;
	}
  std::cout << "running" << std::endl;

  //use linear initialization here
  std::cout << "===start linear initialization ===" << std::endl;
  linear_initialization_test();
  std::cout << "===end linear initialization ===" << std::endl;
  
  //use random initiatization here
  std::cout << "===start random initialization ===" << std::endl;
  random_initialization_test();
  std::cout << "===end random initialization ===" << std::endl;
  return 0;
}

