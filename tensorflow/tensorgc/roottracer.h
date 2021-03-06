#ifndef ROOT_TRACER
#define ROOT_TRACER
#include <set>
#include <mutex>

namespace tensorflow {

//T1 is Tensor
//T2 is TensorBuffer
template <typename T1, typename T2>
class RootTracer{
 public:
  //class method members

  RootTracer();

  // copy constructor
  RootTracer(const RootTracer<T1, T2> &tmp);
  
  //clean all members
  ~RootTracer();
  
  // find
  bool find(T1* root);
  bool find_by_id(int id);
  
  // compare
  bool compare(RootTracer<T1, T2> tmp);

  //Add newtensor to root_set
  void addto_root_set(T1* newtensor);

  //Remove oldtensor from root_set
  void rmfrom_root_set(T1* oldtensor);

  //Put all reachable buffer object references in tracing_set
  //All reachable objects traced from root_set is put in the tracing_set
  //All objects added to the root_set must be TensorBuffer objects
  void start_tracing(std::set<T2*>* tracing_set);

  int getsize_root_set();

  void increment_trace_size(int size);

  void clear_trace_size();

  int get_trace_size();

//  int trace_counter; // how often do we call the start_tracing ???

 private:
  //class field members

  //root_set contains all active tensor obeject references
  //If a new tensor is allocated, we use addto_root_set to add the new tensor to this root_set
  //If an old tensor is deallocated, we use rmfrom_root_set to remove this tensor from this root_set
  //All objects added to the root_set must be Tensor objects
  //std::atomic<std::set<T1*>> root_set;
  std::set<T1*> root_set;

  int trace_size;
  std::mutex mtx;
};//end RootTracer class


}//end tensorflow namespace

#endif
