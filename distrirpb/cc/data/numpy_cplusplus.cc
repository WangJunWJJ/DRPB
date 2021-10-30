//Create by wangjun
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
namespace bp = boost::python;
void reference_contiguous_array(PyObject* in,
        PyArrayObject* in_con,
        double* ptr, int& count)
{
    in_con = PyArray_GETCONTIGUOUS((PyArrayObject*)in);
    ptr = (double*)PyArray_DATA(in_con);
    int num_dim = PyArray_NDIM(in_con);
    npy_intp* pdim = PyArray_DIMS(in_con);
    count = 1;
    for (int i = 0; i < num_dim; i++)
    {
        count *= pdim[i];
    }
}
void dereference(PyObject* o)
{
    Py_DECREF(o);
}
PyObject* entry_square_matrix(PyObject* input_matrix)
{
    // get the input array
    double* ptr;
    int count;
    PyArrayObject* input_contigous_array;
    reference_contiguous_array(input_matrix, input_contigous_array, ptr, count);

    // create the output array
    npy_intp dst_dim[1];
    dst_dim[0] = count;
    PyObject* out_matrix = PyArray_SimpleNew(1, dst_dim, NPY_FLOAT64);
    double* ptr_out;
    PyArrayObject* output_contigous_array;
    reference_contiguous_array(out_matrix, output_contigous_array, ptr_out, count);
    for (int i = 0; i < count; i++)
    {
        ptr_out[i] = ptr[i] * ptr[i];
    }
    dereference((PyObject*)input_contigous_array);
    dereference((PyObject*)output_contigous_array);
    return out_matrix;
}
BOOST_PYTHON_MODULE(_func)
{
    import_array();
    bp::def(square_matrix, entry_square_matrix);
}
