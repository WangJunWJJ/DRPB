//Create by wangjun

#include <iostream>
#include <Python.h>
#include <numpy/arrayobject.h>
using namespace std;

void init_numpy(){

    import_array();
}
int main()
{
    Py_Initialize();
    init_numpy();

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/home/liaojian/Documents/Programming/PythonWorkSpace/CalledByCplusplus/')");

    PyObject *pModule  = nullptr;
    PyObject *pDict    = nullptr;

    pModule  = PyImport_ImportModule("ModuleOne");
    pDict    = PyModule_GetDict(pModule);

    /*Return the List which contains Numpy Array*/
    PyObject *pFuncOne    = PyDict_GetItemString(pDict, "ArrayListReturn");

    PyObject *FuncOneBack = PyObject_CallObject(pFuncOne, nullptr);

    int Index_i = 0, Index_k = 0, Index_m = 0, Index_n = 0;
    if(PyList_Check(FuncOneBack)){

        int SizeOfList = PyList_Size(FuncOneBack);

        for(Index_i = 0; Index_i < SizeOfList; Index_i++){

            PyArrayObject *ListItem = (PyArrayObject *)PyList_GetItem(FuncOneBack, Index_i);

            int Rows = ListItem->dimensions[0], columns = ListItem->dimensions[1];
            cout<<"The "<<Index_i<<"th Array is:"<<endl;
            for(Index_m = 0; Index_m < Rows; Index_m++){

                for(Index_n = 0; Index_n < columns; Index_n++){

                    cout<<*(double *)(ListItem->data + Index_m * ListItem->strides[0] + Index_n * ListItem->strides[1])<<" ";
                }
                cout<<endl;
            }

            Py_DECREF(ListItem);
        }
    }else{

        cout<<"Not a List"<<endl;
    }

    /*Return Integer List and Access to Each Items*/

    cout<<"Integer List Show:"<<endl;
    PyObject *pFuncTwo     = PyDict_GetItemString(pDict, "IntegerListReturn");
    PyObject *FuncTwoBack  = PyObject_CallObject (pFuncTwo, nullptr);

    if(PyList_Check(FuncTwoBack)){

        int SizeOfList = PyList_Size(FuncTwoBack);
        for(Index_i = 0; Index_i < SizeOfList; Index_i++){

            PyObject *ListItem = PyList_GetItem(FuncTwoBack, Index_i);

            int NumOfItems = PyList_Size(ListItem);

            for(Index_k = 0; Index_k < NumOfItems; Index_k++){

                PyObject *Item = PyList_GetItem(ListItem, Index_k);

                cout << PyInt_AsLong(Item) <<" ";

                Py_DECREF(Item);
            }

            Py_DECREF(ListItem);
        }
        cout<<endl;


    }else{

        cout<<"Not a List"<<endl;
    }

    /*Return Float List and Access to Each Items*/

    cout<<"Double List Show:"<<endl;
    PyObject *pFunThree     = PyDict_GetItemString(pDict, "FloatListReturn");
    PyObject *FuncThreeBack = PyObject_CallObject  (pFunThree, nullptr);

    if(PyList_Check(FuncThreeBack)){

        int SizeOfList = PyList_Size(FuncThreeBack);
        for(Index_i = 0; Index_i < SizeOfList; Index_i ++){

            PyObject *ListItem = PyList_GetItem(FuncThreeBack, Index_i);
            int NumOfItems = PyList_Size(ListItem);

            for(Index_k = 0; Index_k < NumOfItems; Index_k++){

                PyObject *Item  = PyList_GetItem(ListItem, Index_k);

                cout<< PyFloat_AsDouble(Item) << " ";

                Py_DECREF(Item);
            }

            Py_DECREF(ListItem);
        }
        cout<<endl;


    }else{

        cout<<"Not a List"<<endl;
    }

    /*Pass by List: Transform an C Array to Python List*/

    double CArray[] = {1.2, 4.5, 6.7, 8.9, 1.5, 0.5};

    PyObject *PyList  = PyList_New(6);
    PyObject *ArgList = PyTuple_New(1);
    for(Index_i = 0; Index_i < PyList_Size(PyList); Index_i++){

        PyList_SetItem(PyList, Index_i, PyFloat_FromDouble(CArray[Index_i]));
    }

    PyObject *pFuncFour = PyDict_GetItemString(pDict, "PassListFromCToPython");
    cout<<"C Array Pass Into The Python List:"<<endl;
    PyTuple_SetItem(ArgList, 0, PyList);
    PyObject_CallObject(pFuncFour, ArgList);

    /*Pass by Python Array: Transform an C Array to Python Array*/

    double CArrays[3][3] = {{1.3, 2.4, 5.6}, {4.5, 7.8, 8.9}, {1.7, 0.4, 0.8}};

    npy_intp Dims[2] = {3, 3};

    PyObject *PyArray  = PyArray_SimpleNewFromData(2, Dims, NPY_DOUBLE, CArrays);
    PyObject *ArgArray = PyTuple_New(1);
    PyTuple_SetItem(ArgArray, 0, PyArray);

    PyObject *pFuncFive = PyDict_GetItemString(pDict, "PassArrayFromCToPython");
    PyObject_CallObject(pFuncFive, ArgArray);

    //Release
    Py_DECREF(pModule);
    Py_DECREF(pDict);
    Py_DECREF(FuncOneBack);
    Py_DECREF(FuncTwoBack);
    Py_DECREF(FuncThreeBack);
    Py_DECREF(PyList);
    Py_DECREF(ArgList);
    Py_DECREF(PyArray);
    Py_DECREF(ArgArray);
    return 0;
}