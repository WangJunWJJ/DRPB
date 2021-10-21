//Create by wangjun
// 2021-10-20
// for the datatype design

#ifndef DISTRIRPB_C_DDM_DATATYPE_H_
#define DISTRIRPB_C_DDM_DATATYPE_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// --------------------------------------------------------------------------
// DDM_DataType holds the type for a scalar value.  E.g., one slot in a tensor.
// The enum values here are identical to corresponding values in types.proto.
typedef enum DDM_DataType {
  DDM_FLOAT = 1,
  DDM_DOUBLE = 2,
  DDM_INT32 = 3,  // Int32 tensors are always in 'host' memory.
  DDM_UINT8 = 4,
  DDM_INT16 = 5,
  DDM_INT8 = 6,
  DDM_STRING = 7,
  DDM_COMPLEX64 = 8,  // Single-precision complex
  DDM_COMPLEX = 8,    // Old identifier kept for API backwards compatibility
  DDM_INT64 = 9,
  DDM_BOOL = 10,
  DDM_QINT8 = 11,     // Quantized int8
  DDM_QUINT8 = 12,    // Quantized uint8
  DDM_QINT32 = 13,    // Quantized int32
  DDM_BFLOAT16 = 14,  // Float32 truncated to 16 bits.  Only for cast ops.
  DDM_QINT16 = 15,    // Quantized int16
  DDM_QUINT16 = 16,   // Quantized uint16
  DDM_UINT16 = 17,
  DDM_COMPLEX128 = 18,  // Double-precision complex
  DDM_HALF = 19,
  DDM_RESOURCE = 20,
  DDM_VARIANT = 21,
  DDM_UINT32 = 22,
  DDM_UINT64 = 23,
} DDM_DataType;

// DDM_DataTypeSize returns the sizeof() for the underlying type corresponding
// to the given DDM_DataType enum value. Returns 0 for variable length types
// (eg. TF_STRING) or on failure.
DDM_CAPI_EXPORT extern size_t DDM_DataTypeSize(DDM_DataType dt);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // DISTRIRPB_C_DDM_DATATYPE_H_