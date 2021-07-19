import numpy as np


def build_norm_instance(num, maximum, f):
  for i in range(num):
    print(round(np.random.uniform(0,maximum),4), file=f)
    
num = 10000
for i in [4,8]:
  with open('data/psk_' + str(i) +  '_' + str(num) + '.txt', 'w') as f:
    build_norm_instance(num,i,f)
