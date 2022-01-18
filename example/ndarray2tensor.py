'''
Author: Jun Wang
Date: 2022-01-13 17:47:37
LastEditTime: 2022-01-15 17:41:22
LastEditors: Please set LastEditors
FilePath: /DRPB/example/ndarray2tensot.py
'''
import numpy as np
import torch

a=np.array([1,2,3,4,5,6,7,8])
print(a)
print(a.dtype)
a_tensor=torch.from_numpy(a)
print(a_tensor)


print('--------------------------')

b = [[1,2],[3,4],[5,6],[7,8]]
print(b)
bb=np.array(b)
print(bb)
print(bb.dtype)
b_tensor=torch.from_numpy(bb)
print(b_tensor)








