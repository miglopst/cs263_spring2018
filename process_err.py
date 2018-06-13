#This script is used to profile totoal buffer memory in Tensor allocation and Buffer allication
import re
import matplotlib.pyplot as plt
filepath = './err.txt'  
with open(filepath) as fp:
  content = fp.readlines()
content = [x.strip() for x in content]

ct1=0
ct2=0
ct3=0
ct4=0
ct5=0
ct6=0
cct=0
mct=0
dt=0
eff_ct=0
bufaloc=0
bufdealoc=0
cur_buf=0
buf_status = []
mem_status = []
number = []
time = 0
time_base = 0
time_axis = []
flag = 0

#to profile memory footprint
buf_mem=[]
tensor_buf_mem=[]

for item in content:
  if "[Peng]tensorflow/core/framework/tensor.cc:Buffer()" in item:
    bufaloc = bufaloc + 1
    cur_buf = cur_buf + 1
    number = re.findall(r'\d+',item)
    #The useful memory footprint information is in number[9]
    if (flag==0):
      time_base = (int(number[4])*60+int(number[5]))*1000000+int(number[6])
      time = 0
      flag = 1
    else:
      time = (int(number[4])*60+int(number[5]))*1000000+int(number[6]) - time_base
    buf_mem.append(int(number[9]))
  if "[Peng]tensorflow/core/framework/tensor.cc:~Buffer()" in item:
    bufdealoc = bufdealoc + 1
    cur_buf = cur_buf - 1
    number = re.findall(r'\d+',item)
    #The useful memory footprint information is in number[9]
    if (flag==0):
      time_base = (int(number[4])*60+int(number[5]))*1000000+int(number[6])
      time = 0
      flag = 1
    else:
      time = (int(number[4])*60+int(number[5]))*1000000+int(number[6]) - time_base
    buf_mem.append(int(number[8]))
  elif "Tensor constructor 2" in item:
    ct2 = ct2 + 1
    eff_ct = eff_ct + 1
    number = re.findall(r'\d+',item)
    #The useful memory footprint information is in number[9]
    if (flag==0):
      time_base = (int(number[4])*60+int(number[5]))*1000000+int(number[6])
      time = 0
      flag = 1
    else:
      time = (int(number[4])*60+int(number[5]))*1000000+int(number[6]) - time_base
    tensor_buf_mem.append(int(number[9]))
  elif "Tensor constructor 3" in item:
    ct3 = ct3 + 1
    eff_ct = eff_ct + 1
    number = re.findall(r'\d+',item)
    if (flag==0):
      time_base = (int(number[4])*60+int(number[5]))*1000000+int(number[6])
      time = 0
      flag = 1
    else:
      time = (int(number[4])*60+int(number[5]))*1000000+int(number[6]) - time_base
    tensor_buf_mem.append(int(number[9]))
  elif "Tensor constructor 4" in item:
    ct4 = ct4 + 1
    eff_ct = eff_ct + 1
    number = re.findall(r'\d+',item)
    if (flag==0):
      time_base = (int(number[4])*60+int(number[5]))*1000000+int(number[6])
      time = 0
      flag = 1
    else:
      time = (int(number[4])*60+int(number[5]))*1000000+int(number[6]) - time_base
    tensor_buf_mem.append(int(number[9]))
  elif "Tensor constructor 5" in item:
    ct5 = ct5 + 1
    eff_ct = eff_ct + 1
    number = re.findall(r'\d+',item)
    if (flag==0):
      time_base = (int(number[4])*60+int(number[5]))*1000000+int(number[6])
      time = 0
      flag = 1
    else:
      time = (int(number[4])*60+int(number[5]))*1000000+int(number[6]) - time_base
    tensor_buf_mem.append(int(number[9]))
  elif "[Peng]tensorflow/core/framework/tensor.cc:Tensor deconstructor" in item:
    number = re.findall(r'\d+',item)
    if (flag==0):
      time_base = (int(number[4])*60+int(number[5]))*1000000+int(number[6])
      time = 0
      flag = 1
    else:
      time = (int(number[4])*60+int(number[5]))*1000000+int(number[6]) - time_base
    tensor_buf_mem.append(int(number[8]))
  mem_status.append(eff_ct)
  time_axis.append(time)
  buf_status.append(cur_buf)

tensor_buf_mem = [x/(1024*1024) for x in tensor_buf_mem]
plt.plot(tensor_buf_mem)
plt.ylabel('Buffer Memory (MB)')
plt.xlabel('step')
#plt.plot(buf_mem)
plt.show()
#plt.plot(time_axis,mem_status)
#plt.plot(time_axis,buf_status)

#plt.ylabel('Number of Tensor (blue) / Buffer (green)')
#plt.xlabel('step')
#plt.show()

#print "constructor 2 = " + str(ct2)
#print "constructor 3 = " + str(ct3)
#print "constructor 4 = " + str(ct4)
#print "constructor 5 = " + str(ct5)

