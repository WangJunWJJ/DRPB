//Create by wangjun
//2021-10-20
//
#include "tensorflow/c/tf_datatype.h"

#include "tensorflow/core/framework/types.h"

size_t DDM_DataTypeSize(TF_DataType dt) {
  return static_cast<size_t>(
      tensorflow::DataTypeSize(static_cast<tensorflow::DataType>(dt)));
}