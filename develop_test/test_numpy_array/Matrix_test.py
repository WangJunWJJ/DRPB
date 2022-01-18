'''
Author: your name
Date: 2022-01-05 16:35:40
LastEditTime: 2022-01-15 19:17:08
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /DRPC/src/Test_develop/Matrix_test.py
'''
import demo as numpy_demo2
import numpy as np


var1 = numpy_demo2.add_arrays_1d(np.array([1, 3, 5, 7, 9]),
                                 np.array([2, 4, 6, 8, 10]))
print('-'*50)
print('var1', var1)

var2 = numpy_demo2.add_arrays_2d(np.array(range(0,16)).reshape([4, 4]),
                                 np.array(range(20,36)).reshape([4, 4]))
print('-'*50)
print('var2', var2)

input1 = np.array(range(0, 48)).reshape([4, 4, 3])
input2 = np.array(range(50, 50+48)).reshape([4, 4, 3])
var3 = numpy_demo2.add_arrays_3d(input1,
                                 input2)
print('-'*50)
print('var3', var3)


input_test = np.array(range(0, 1151)).reshape([4, 4, 3, 4, 6])

numpy_demo2.test_not_sure_dim(input_test)