/*
 * @Author: your name
 * @Date: 2022-01-06 17:27:46
 * @LastEditTime: 2022-01-06 21:15:22
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: /DRPB/third_party/example.cc
 */
#include<pybind11/embed.h>
#include<iostream>
#include"Python.h"

#include<numpy.h>
#include<numpy/ndarrayobject.h>
#include<python3.8/Python.h>
namespace py = pybind11;

int main(){
	py::scoped_interpreter python;
	py::module t = py::module::import("example");
	t.attr("add")(1,2);
//	printf("CC add:%d",y);
	return 0;
}