#ifndef BUF_TRACER
#define BUF_TRACER
#include <set>

namespace tensorflow {

//T is TensorBuffer
template <typename T>
class BufTracer{
 public:
  //class method members

  BufTracer(int thresh = 1024);

  //clean all members
  ~BufTracer();

  //All newly allocated TensorBuffer object references are added in this buffer_set
  //There is no remove buffer functions, since this is managed by GC
  void addto_buffer_set(T* newbuffer);
  void rmfrom_buffer_set(T* oldbuffer);

  //Move all unreachable buffer objects from buffer_set to garbage_set
  void mark_mv_garbage_set();

  //Free garbage_set
  //For each garbage, call unref() to tigger ~Buffer() and Deallocate(), DeallocateRaw()
  void free_garbage_set();

  std::set<T*>* get_tracing_set();

  int get_tracing_set_size();

  int get_buffer_set_size();
  
  int get_thresh();

 private:

  //All reachable objects traced from root_set is put in the tracing_set
  //All objects added to the root_set must be TensorBuffer objects
  std::set<T*> tracing_set;

  //class field members

  //All allocated TensorBuffer objects are added to this set
  //All objects added to the root_set must be TensorBuffer objects
  std::set<T*> buffer_set;

  //All unreachable objects during tracing are added to this set
  //All objects added to the root_set must be TensorBuffer objects
  std::set<T*> garbage_set;
  
  int tracing_set_size;
  
  int tracing_thresh;

};//end BufTracer class


}//end tensorflow namespace
#endif
