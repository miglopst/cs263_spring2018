import re
#import matplotlib
#matplotlib.use('GTK')
import matplotlib.pyplot as plt
filepath = './log.txt'
with open(filepath) as fp:
  content = fp.readlines()
content = [x.strip() for x in content]
req_alloc_size = 0
avg_req_alloc_size = 0
actual_alloc_size = 0
dealloc_size = 0
current_size = 0
free_inuse = 0
alloc_inuse = 0
mem_status = []
i = 0
j = 0
cnt = 0
in_frag = 0
in_frag_cnt = 0
for item in content:
  if "[Peng][bfc_allocator.cc]-allocate" in item:
    for num in re.findall(r'\d+',item):
      req_alloc_size = req_alloc_size + int(num)
    i = i + 1
  if "[Peng][bfc_allocator.cc]-actual_allocate" in item:
    for num in re.findall(r'\d+',item):
      actual_alloc_size = actual_alloc_size + int(num)
      current_size = current_size + int(num)
    j = j + 1
  if "[Peng][bfc_allocator.cc][FreeAndMaybeCoalesce]-deallocate" in item:
    for num in re.findall(r'\d+',item):
      dealloc_size = dealloc_size + int(num)
      current_size = current_size - int(num)
  if "[Peng][bfc_allocator.cc][FreeAndMaybeCoalesce]-bytes_in_use" in item:
    for num in re.findall(r'\d+',item):
      free_inuse = int(num)
      #mem_status.append(free_inuse)
  if "[Peng][bfc_allocator.cc]-bytes_in_use" in item:
    for num in re.findall(r'\d+',item):
      alloc_inuse = int(num)
      #mem_status.append(alloc_inuse)
  mem_status.append(current_size)
  if "step" in item:
    temp_in_frag = 100-req_alloc_size/float(actual_alloc_size)*100
    if (cnt > 0):
      if (temp_in_frag>0 and temp_in_frag<100):
        in_frag = in_frag + temp_in_frag
        avg_req_alloc_size = avg_req_alloc_size + req_alloc_size
        in_frag_cnt = in_frag_cnt + 1
    cnt = cnt + 1
    print "dealloc size = "+str(dealloc_size)+" bytes"
    print "request alloc size = "+str(req_alloc_size)+" bytes"
    print "actual alloc size = "+str(actual_alloc_size)+" bytes"
    #print "internal fragmentation ratio (%) = " + str(temp_in_frag)
    #print "actual_alloc_size / dealloc_size (%) = " + str(actual_alloc_size/float(dealloc_size)*100)
    dealloc_size = 0
    req_alloc_size = 0
    actual_alloc_size = 0
print "avg request allocation size (bytes) = " + str(avg_req_alloc_size/float(in_frag_cnt))
print "avg internal fragmentation ratio (%) = " + str(in_frag/in_frag_cnt)
base = 1024*1024
mem_status = [x/base for x in mem_status]
plt.plot(mem_status)
plt.ylabel('Mbytes')
plt.show()
mem_status.append(current_size)

