#ifndef _MEXUTILS_H
#define _MEXUTILS_H

#include "mex.h"

#include<string>
#include<vector>
#include<map>

// get size / dimensions
void doubleMatrixSize(const mxArray* input, mwSize& m, mwSize& n);

void doubleArraySize(const mxArray* input, std::vector<mwSize>& size);

// read
void readDoubleScalar(const mxArray* input, double& output);

void readLogicalScalar(const mxArray* input, bool& output);

void readString(const mxArray* input, std::string& output);

void readDoubleVector(const mxArray* input, std::vector<double>& output);

void readDoubleNDVectorset(const mxArray* input, const mwSize& nd, std::vector<double>& output);

void readDoubleMatrix(const mxArray* input, std::vector<double>& output, mwSize& m, mwSize& n);
void readDoubleMatrix(const mxArray* input, double*& output, mwSize& m, mwSize& n);

void readLogicalVector(const mxArray* input, std::vector<mxLogical>& output);

void readLogicalArray(const mxArray* input, const std::vector<mwSize>& size, bool allowempty,
        std::vector<bool>& output, std::vector<mwSize>& outputsize);

void readDoubleArray(const mxArray* input, const std::vector<mwSize>& size, bool allowempty,
        std::vector<double>& output, std::vector<mwSize>& outputsize);

void readCellArray(const mxArray* input, std::vector<const mxArray*>& output);

void readCellVector(const mxArray* input, std::vector<const mxArray*>& output);

void readCellMatrix(const mxArray* input, std::vector<const mxArray*>& output, mwSize& m, mwSize& n);

void readStringCellArray(const mxArray* input, std::vector<std::string>& output);

void readDoubleVectorCellArray(const mxArray* input, std::vector<std::vector<double> >& output);

void readDoubleMatrixCellArray(const mxArray* input, std::vector<std::vector<double> >& output, std::vector<mwSize>& m, std::vector<mwSize>& n);

void readDoubleVectorCellMatrix(const mxArray* input, std::vector<std::vector<double> >& output, mwSize& m, mwSize& n);

void readDoubleNDVectorsetCellArray(const mxArray* input, const mwSize& nd, std::vector<std::vector<double> >& output);

void readDoubleNDVectorsetCellMatrix(const mxArray* input, const mwSize& nd, std::vector<std::vector<double> >& output, mwSize& m, mwSize& n);

void readNameValuePairs(mwSize ninput, const mxArray** input, std::map<std::string,const mxArray*>& output);

std::map<std::string,const mxArray*>::iterator findName(
        std::map<std::string,const mxArray*>& nvpairs, const std::string& name);

void getNamedLogicalArray(std::map<std::string,const mxArray*>& nvpairs,
        const std::string& name, const std::vector<mwSize>& size,
        bool allowempty, bool removefromlist,
        std::vector<bool>& output, std::vector<mwSize>& outputsize);

void getNamedDoubleArray(std::map<std::string,const mxArray*>& nvpairs,
        const std::string& name, const std::vector<mwSize>& size,
        bool allowempty, bool removefromlist,
        std::vector<double>& output, std::vector<mwSize>& outputsize);

void getNamedString(std::map<std::string,const mxArray*>& nvpairs,
        const std::string& name, bool removefromlist, std::string& output);

void getNamedLogicalScalar(std::map<std::string,const mxArray*>& nvpairs,
        const std::string& name, bool removefromlist, bool& output);

void getNamedDoubleScalar(std::map<std::string,const mxArray*>& nvpairs,
        const std::string& name, bool removefromlist, double& output);

// write
void writeString(const std::string& input, mxArray*& output);

void writeStringArray(const std::vector<std::string>& input, mxArray*& output, mwSize m, mwSize n);

void writeLogicalMatrix(const std::vector<bool>& input, mxArray*& output, mwSize m, mwSize n);

template<typename T>
void writeDoubleMatrix(const std::vector<T>& input, mxArray*& output, mwSize m, mwSize n)
{
    if (m*n != input.size()) {
        mexErrMsgTxt("Input size does not match requested size.");
    }
    
    output = mxCreateDoubleMatrix(m,n,mxREAL);
    std::copy(input.begin(),input.end(),mxGetPr(output));
};

template<typename T>
void writeDoubleMatrix(const T* input, mxArray*& output, mwSize m, mwSize n)
{
    output = mxCreateDoubleMatrix(m,n,mxREAL);
    std::copy(input,input + m*n,mxGetPr(output));
};

void writeCellArray(const std::vector<mxArray*>& input, mxArray*& output, mwSize m, mwSize n);

// nrows: number of rows in each cell
template<typename T>
void writeDoubleRowCellArray(const std::vector<std::vector<T> >& input, mxArray*& output, mwSize m, mwSize n, mwSize nrows=1)
{
    std::vector<mxArray*> cells(input.size(),NULL);
    typename std::vector<T>::const_iterator it;
    for (auto it=input.begin(); it!=input.end(); it++) {
        if (it->size() % nrows != 0) {
            mexErrMsgTxt("One of the cells does not fit in the requested dimensions.");
        }
        mxArray* cell;
        writeDoubleMatrix(*it, cell,nrows,it->size()/nrows);
        cells[it-input.begin()] = cell;
    }
    
    writeCellArray(cells, output, m,n);
};

// ncols: number of columns in each cell
template<typename T>
void writeDoubleColumnCellArray(const std::vector<std::vector<T> >& input, mxArray*& output, mwSize m, mwSize n, mwSize ncols=1)
{
    std::vector<mxArray*> cells(input.size(),NULL);
    typename std::vector<T>::const_iterator it;
    for (auto it=input.begin(); it!=input.end(); it++) {
        if (it->size() % ncols != 0) {
            mexErrMsgTxt("One of the cells does not fit in the requested dimensions.");
        }
        mxArray* cell;
        writeDoubleMatrix(*it, cell,it->size()/ncols,ncols);
        cells[it-input.begin()] = cell;
    }
    
    writeCellArray(cells, output, m,n);
};

#endif // _MEXUTILS_H
