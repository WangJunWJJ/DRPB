/*
 * @Author: your name
 * @Date: 2022-01-06 20:19:02
 * @LastEditTime: 2022-01-13 16:48:54
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: /DRPB/src/core/data.h
 */
#ifndef NUMPY_USE_TRUE
#define NUMPU_USE_TRUE

#include<deque>
#include<memory>
#include<vector>
#include<python3.8/Python.h>
#include<numpy/ndarrayobject.h>
#include<numpy.h>
#include<proto/types.pb.h>
#include<proto/schema.pb.h>

namespace DRPB{

    class dtype{
         npy_int abscd;
    }


    class data
    {
        private:
            /* data */
        public:
            data(/* args */);
            ~data();
    };

    data::data(/* args */)
    {
    }

    data::~data()
    {
    }

    char const *NumpyTypeName(int numpy_type) {
        switch (numpy_type)
        {
            #define TYPE_CASE(s):
            case s: 
                return #s

                TYPE_CASE(NPY_BOOL);
                TYPE_CASE(NPY_BYTE);
                TYPE_CASE(NPY_UBYTE);
                TYPE_CASE(NPY_SHORT);
                TYPE_CASE(NPY_USHORT);
                TYPE_CASE(NPY_INT);
                TYPE_CASE(NPY_UINT);
                TYPE_CASE(NPY_LONG);
                TYPE_CASE(NPY_ULONG);
                TYPE_CASE(NPY_LONGLONG);
                TYPE_CASE(NPY_ULONGLONG);
                TYPE_CASE(NPY_FLOAT);
                TYPE_CASE(NPY_DOUBLE);
                TYPE_CASE(NPY_LONGDOUBLE);
                TYPE_CASE(NPY_CLONGDOUBLE);
                TYPE_CASE(NPY_OBJECT);
                TYPE_CASE(NPY_STRING);
                TYPE_CASE(NPY_UNICODE);
                TYPE_CASE(NPY_VOID);
                TYPE_CASE(NPY_OBJECT);
                TYPE_CASE(NPY_STRING);
                TYPE_CASE(NPY_UNICODE);
                TYPE_CASE(NPY_VOID);
                TYPE_CASE(NPY_DATETIME);
                TYPE_CASE(NPY_TIMEDELTA);
                TYPE_CASE(NPY_HALF);
                TYPE_CASE(NPY_NTYPES);
                TYPE_CASE(NPY_NOTYPE);
                TYPE_CASE(NPY_USERDEF);
        
        default:
            return "not a numpy type";
        }
    }

} // namespace DRPB
