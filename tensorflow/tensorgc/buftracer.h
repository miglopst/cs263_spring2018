#ifndef BUF_TRACER
#define BUF_TRACER
#include <set>

namespace tensorflow {

//T is TensorBuffer
template <typename T>
class BufTracer{
 public:
  //class method members

  BufTracer();

  //clean all members
  ~BufTracer();

  //All newly allocated TensorBuffer object references are added in this buffer_set
  //There is no remove buffer functions, since this is managed by GC
  void addto_buffer_set(T* newbuffer);

  //Move all unreachable buffer objects from buffer_set to garbage_set
  void mark_mv_garbage_set();

  //Free garbage_set
  //For each garbage, call unref() to tigger ~Buffer() and Deallocate(), DeallocateRaw()
  void free_garbage_set();

  std::set<T*>* get_tracing_set();

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

};//end BufTracer class


}//end tensorflow namespace
#endif
