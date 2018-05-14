#include <cstdlib>
#include <string>

int debug_flag(){
  if(const char* env_p = std::getenv("DEBUG_FLAG")){
    return atoi(env_p);
  }
  else
    return 0;
}
