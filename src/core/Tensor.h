/*
 * @Author: your name
 * @Date: 2022-01-06 20:33:15
 * @LastEditTime: 2022-01-06 21:14:57
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: /DRPB/src/core/Tensor.h
 */
#ifndef NUMPY_TENSOR_USE_H_
#define NUMPY_TENSOR_USE_H_

#include<deque>
#include<memory>
#include<vector>
#include<python3.8/Python.h>
#include<numpy/ndarrayobject.h>
#include<numpy.h>

namespace DRPB{
    class Tensor
    {
    private:
        /* data */
    public:
        Tensor(/* args */);
        ~Tensor();
    };
    
    Tensor::Tensor(/* args */)
    {
    }
    
    Tensor::~Tensor()
    {
    }
    

}// namespace DRPB