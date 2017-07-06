#include "mex.h"

#include<stdint.h>
#include<string>
#include<vector>
#include<map>

#include "mexutils.h"

void readDoubleScalar(const mxArray* input, double& output)
{
    if (!mxIsClass(input,"double")) {
        mexErrMsgTxt("Input is not double.");
    }
    if (mxGetNumberOfDimensions(input) != 2) {
        mexErrMsgTxt("Input number of dimensions for double scalar.");
    }
    if (mxGetM(input) != 1 || mxGetN(input) != 1) {
        mexErrMsgTxt("Input is not a scalar.");
    }

    output = mxGetScalar(input);
}

void readLogicalScalar(const mxArray* input, bool& output)
{
    if (!mxIsClass(input,"logical")) {
        mexErrMsgTxt("Input is not logical.");
    }
    if (mxGetNumberOfDimensions(input) != 2) {
        mexErrMsgTxt("Input number of dimensions for logical scalar.");
    }
    if (mxGetM(input) != 1 || mxGetN(input) != 1) {
        mexErrMsgTxt("Input is not a scalar.");
    }

    output = (bool) *mxGetLogicals(input);
}

void readString(const mxArray* input, std::string& output)
{
    if (!mxIsClass(input,"char")) {
        mexErrMsgTxt("Input is not a string.");
    }
    if (mxGetNumberOfDimensions(input) != 2) {
        mexErrMsgTxt("Invalid number of dimensions for string.");
    }
    if (mxIsEmpty(input)) {
        output.clear();
        return;
    }
    
    if (mxGetM(input) != 1) {
        mexErrMsgTxt("Invalid input dimensions for string.");
    }

    char* name = mxArrayToString(input);    
    output.assign(name);
    mxFree(name);
}

void readDoubleVector(const mxArray* input, std::vector<double>& output)
{
    if (!mxIsClass(input,"double")) {
        mexErrMsgTxt("Type is not double.");
    }
    if (mxGetNumberOfDimensions(input) != 2 || mxGetN(input) != 1) {
        mexErrMsgTxt("Input is not a vector.");
    }
    
    output.assign(mxGetPr(input),mxGetPr(input)+mxGetNumberOfElements(input));
}

void readDoubleNDVectorset(const mxArray* input, const mwSize& nd, std::vector<double>& output)
{
    if (!mxIsClass(input,"double")) {
        mexErrMsgTxt("Type is not double.");
    }
    if (mxGetNumberOfDimensions(input) != 2 || mxGetM(input) != nd) {
        mexErrMsgTxt("Input is not an ND vector set.");
    }
    
    output.assign(mxGetPr(input),mxGetPr(input)+mxGetNumberOfElements(input));
}

void doubleMatrixSize(const mxArray* input, mwSize& m, mwSize& n)
{
    if (!mxIsClass(input,"double")) {
        mexErrMsgTxt("Type is not double.");
    }
    if (mxGetNumberOfDimensions(input) != 2) {
        mexErrMsgTxt("Invalid number of dimensions for double matrix.");
    }
    m = mxGetM(input);
    n = mxGetN(input);
}

void readDoubleMatrix(const mxArray* input, double*& output, mwSize& m, mwSize& n)
{
    if (!mxIsClass(input,"double")) {
        mexErrMsgTxt("Type is not double.");
    }
    if (mxGetNumberOfDimensions(input) != 2) {
        mexErrMsgTxt("Invalid number of dimensions for double matrix.");
    }
    m = mxGetM(input);
    n = mxGetN(input);
    
    output = mxGetPr(input);
}

void readDoubleMatrix(const mxArray* input, std::vector<double>& output, mwSize& m, mwSize& n)
{
    double* optr = NULL;
    readDoubleMatrix(input, optr, m, n);
    
    output.assign(optr,optr+mxGetNumberOfElements(input));
}

void readLogicalVector(const mxArray* input, std::vector<mxLogical>& output)
{
    if (!mxIsClass(input,"logical")) {
        mexErrMsgTxt("Type is not logical.");
    }
    if (mxGetNumberOfDimensions(input) != 2) {
        mexErrMsgTxt("Invalid number of dimensions for double vector.");
    }
    if (mxGetN(input) != 1) {
        mexErrMsgTxt("Invalid size for logical vector.");
    }
    
    output.assign(mxGetLogicals(input),mxGetLogicals(input)+mxGetM(input));
}

void readCellArray(const mxArray* input, std::vector<const mxArray*>& output)
{
    if (!mxIsClass(input,"cell")) {
        mexErrMsgTxt("Type is not cell.");
    }

    mwSize nelements = mxGetNumberOfElements(input);
    output.resize(nelements);
    for (mwIndex i=0; i<nelements; i++) {
        output[i] = mxGetCell(input,i);
    }
}

void readCellVector(const mxArray* input, std::vector<const mxArray*>& output)
{
    if (mxGetNumberOfDimensions(input) != 2 || mxGetN(input) != 1) {
        mexErrMsgTxt("Input is not a vector.");
    }
    
    readCellArray(input,output);
}

void readCellMatrix(const mxArray* input, std::vector<const mxArray*>& output, mwSize& m, mwSize& n)
{
    if (mxGetNumberOfDimensions(input) != 2) {
        mexErrMsgTxt("Input is not a matrix.");
    }
    
    m = mxGetM(input);
    n = mxGetN(input);
    
    readCellArray(input,output);
}

void readStringCellArray(const mxArray* input, std::vector<std::string>& output)
{
    if (!mxIsClass(input,"cell")) {
        mexErrMsgTxt("The string array has to be a cell array.");
    }
    
    mwSize nelements = mxGetNumberOfElements(input);
    output.resize(nelements,"");
    for (mwIndex i=0; i<nelements; i++) {
        const mxArray* cell = mxGetCell(input,i);
        if (cell != NULL) {
            readString(cell,output[i]);
        }
    }
}

void readDoubleVectorCellArray(const mxArray* input, std::vector<std::vector<double> >& output)
{
    if (!mxIsClass(input,"cell")) {
        mexErrMsgTxt("Type is not cell.");
    }

    mwSize nelements = mxGetNumberOfElements(input);
    output.resize(nelements);
    for (mwIndex i=0; i<nelements; i++) {
        const mxArray* cell = mxGetCell(input,i);
        if (cell != NULL) { // && mxGetN(cell) >= 1) {
            readDoubleVector(cell,output[i]);
        }
    }
}

void readDoubleMatrixCellArray(const mxArray* input, std::vector<std::vector<double> >& output, std::vector<mwSize>& m, std::vector<mwSize>& n)
{
    if (!mxIsClass(input,"cell")) {
        mexErrMsgTxt("Type is not cell.");
    }

    mwSize nelements = mxGetNumberOfElements(input);
    output.resize(nelements);
    m.resize(nelements);
    n.resize(nelements);
    for (mwIndex i=0; i<nelements; i++) {
        const mxArray* cell = mxGetCell(input,i);
        if (cell != NULL) { // && mxGetN(cell) >= 1) {
            readDoubleMatrix(cell,output[i],m[i],n[i]);
        }
    }
}

void readDoubleVectorCellMatrix(const mxArray* input, std::vector<std::vector<double> >& output, mwSize& m, mwSize& n)
{
    if (mxGetNumberOfDimensions(input) != 2) {
        mexErrMsgTxt("Input is not a matrix.");
    }
    m = mxGetM(input);
    n = mxGetN(input);

    readDoubleVectorCellArray(input,output);
}

void readDoubleNDVectorsetCellArray(const mxArray* input, const mwSize& nd, std::vector<std::vector<double> >& output)
{
    if (!mxIsClass(input,"cell")) {
        mexErrMsgTxt("Type is not cell.");
    }

    mwSize nelements = mxGetNumberOfElements(input);
    output.resize(nelements);
    for (mwIndex i=0; i<nelements; i++) {
        const mxArray* cell = mxGetCell(input,i);
        if (cell != NULL) { // && mxGetN(cell) >= 1) {
            readDoubleNDVectorset(cell,nd,output[i]);
        }
    }
}

void readDoubleNDVectorsetCellMatrix(const mxArray* input, const mwSize& nd, std::vector<std::vector<double> >& output, mwSize& m, mwSize& n)
{
    if (mxGetNumberOfDimensions(input) != 2) {
        mexErrMsgTxt("Input is not a matrix.");
    }
    m = mxGetM(input);
    n = mxGetN(input);

    readDoubleNDVectorsetCellArray(input,nd,output);
}

void readNameValuePairs(mwSize ninput, const mxArray** input,
        std::map<std::string,const mxArray*>& output)
{
    mwIndex nextinputind = 0;
    // name / value pairs
    while (ninput >= nextinputind+1) {
        if (!mxIsClass(input[nextinputind],"char")) {
            mexErrMsgTxt("Invalid input arguments, expected parameter name / value pairs");
        }
        if (mxGetNumberOfDimensions(input[nextinputind]) != 2) {
            mexErrMsgTxt("Some inputs have invalid number of dimensions.");
        }
        if (mxGetM(input[nextinputind]) != 1) {
            mexErrMsgTxt("Invalid parameter name.");
        }
        if (ninput < nextinputind+2) {
            mexErrMsgTxt("Invalid specification of name/value pairs, value is missing.");
        }

        char* name = mxArrayToString(input[nextinputind]);
        output[std::string(name)] = input[nextinputind+1];
        mxFree(name);
        name = NULL;
        
        nextinputind+=2;
    }
}

std::map<std::string,const mxArray*>::iterator findName(
        std::map<std::string,const mxArray*>& nvpairs, const std::string& name)
{
    return nvpairs.find(name);
}

void getNamedLogicalArray(std::map<std::string,const mxArray*>& nvpairs,
        const std::string& name, const std::vector<mwSize>& size,
        bool allowempty, bool removefromlist,
        std::vector<bool>& output, std::vector<mwSize>& outputsize)
{
    if (size.size()==1) {
        mexErrMsgTxt("Must always pass at least two elements in size.");
    }
    
    std::map<std::string,const mxArray*>::iterator pit = nvpairs.find(name);
    if (pit != nvpairs.end()) {
        const mxArray* param = pit->second;
        
        readLogicalArray(param,size,allowempty,output,outputsize);

        if (removefromlist) {
            nvpairs.erase(pit);
        }
    } else {
        std::string errMsg = "Name '"+name+"' not found.";
        mexErrMsgTxt(errMsg.c_str());
    }
}

void getNamedDoubleArray(std::map<std::string,const mxArray*>& nvpairs,
        const std::string& name, const std::vector<mwSize>& size,
        bool allowempty, bool removefromlist,
        std::vector<double>& output, std::vector<mwSize>& outputsize)
{
    if (size.size()==1) {
        mexErrMsgTxt("Must always pass at least two elements in size.");
    }
    
    std::map<std::string,const mxArray*>::iterator pit = nvpairs.find(name);
    if (pit != nvpairs.end()) {
        const mxArray* param = pit->second;
        
        readDoubleArray(param,size,allowempty,output,outputsize);

        if (removefromlist) {
            nvpairs.erase(pit);
        }
    } else {
        std::string errMsg = "Name '"+name+"' not found.";
        mexErrMsgTxt(errMsg.c_str());
    }
}

void getNamedString(std::map<std::string,const mxArray*>& nvpairs,
        const std::string& name, bool removefromlist, std::string& output)
{
    std::map<std::string,const mxArray*>::iterator pit = nvpairs.find(name);
    if (pit != nvpairs.end()) {
        const mxArray* param = pit->second;
        
        readString(param,output);

        if (removefromlist) {
            nvpairs.erase(pit);
        }
    } else {
        std::string errMsg = "Name '"+name+"' not found.";
        mexErrMsgTxt(errMsg.c_str());
    }
}

void getNamedLogicalScalar(std::map<std::string,const mxArray*>& nvpairs,
        const std::string& name, bool removefromlist, bool& output)
{
    std::map<std::string,const mxArray*>::iterator pit = nvpairs.find(name);
    if (pit != nvpairs.end()) {
        const mxArray* param = pit->second;
        
        readLogicalScalar(param,output);

        if (removefromlist) {
            nvpairs.erase(pit);
        }
    } else {
        std::string errMsg = "Name '"+name+"' not found.";
        mexErrMsgTxt(errMsg.c_str());
    }
}

void getNamedDoubleScalar(std::map<std::string,const mxArray*>& nvpairs,
        const std::string& name, bool removefromlist, double& output)
{
    std::map<std::string,const mxArray*>::iterator pit = nvpairs.find(name);
    if (pit != nvpairs.end()) {
        const mxArray* param = pit->second;
        
        readDoubleScalar(param,output);

        if (removefromlist) {
            nvpairs.erase(pit);
        }
    } else {
        std::string errMsg = "Name '"+name+"' not found.";
        mexErrMsgTxt(errMsg.c_str());
    }
}

void readLogicalArray(const mxArray* input, const std::vector<mwSize>& size, bool allowempty,
        std::vector<bool>& output, std::vector<mwSize>& outputsize)
{
    if (!mxIsClass(input,"logical")) {
        mexErrMsgTxt("Invalid value type, must be logical.");
    }
    mwSize ndims = mxGetNumberOfDimensions(input);
    const mwSize *dims = mxGetDimensions(input);
    if (!size.empty() && !(allowempty && mxIsEmpty(input))) {
        if (ndims != size.size()) {
            mexErrMsgTxt("Input does not match requested number of dimensions.");
        }
        for (mwIndex i=0; i<size.size(); i++) {
            if (dims[i] != size[i]) {
                mexErrMsgTxt("Input does not match requested size.");
            }
        }
    }
    
    output.assign(mxGetLogicals(input),mxGetLogicals(input)+mxGetNumberOfElements(input));
    outputsize.assign(dims,dims+ndims);
}

void doubleArraySize(const mxArray* input, std::vector<mwSize>& size)
{
    if (!mxIsClass(input,"double")) {
        mexErrMsgTxt("Invalid value type, must be double.");
    }
    mwSize ndims = mxGetNumberOfDimensions(input);
    const mwSize *dims = mxGetDimensions(input);
    
    size.assign(dims,dims+ndims);
}

void readDoubleArray(const mxArray* input, const std::vector<mwSize>& size, bool allowempty,
        std::vector<double>& output, std::vector<mwSize>& outputsize)
{
    if (!mxIsClass(input,"double")) {
        mexErrMsgTxt("Invalid value type, must be double.");
    }
    mwSize ndims = mxGetNumberOfDimensions(input);
    const mwSize *dims = mxGetDimensions(input);
    if (!size.empty() && !(allowempty && mxIsEmpty(input))) {
        if (ndims != size.size()) {
            mexErrMsgTxt("Input does not match requested number of dimensions.");
        }
        for (mwIndex i=0; i<size.size(); i++) {
            if (dims[i] != size[i]) {
                mexErrMsgTxt("Input does not match requested size.");
            }
        }
    }
    
    output.assign(mxGetPr(input),mxGetPr(input)+mxGetNumberOfElements(input));
    outputsize.assign(dims,dims+ndims);
}

void writeString(const std::string& input, mxArray*& output)
{
    output = mxCreateString(input.c_str());
}

void writeStringArray(const std::vector<std::string>& input, mxArray*& output, mwSize m, mwSize n)
{
//    if (m*n != input.size()) {
//        mexErrMsgTxt("Input size does not match requested size.");
//    }
//    output = mxCreateCellMatrix(m,n);
//    
//    std::vector<std::string>::const_iterator it;
//    for (it=input.begin(); it!=input.end(); it++) {
//        mxArray* cell;
//        writeString(*it, cell);
//        mxSetCell(output,it-input.begin(),cell);
//    }
//
//    if (m*n != input.size()) {
//        mexErrMsgTxt("Input size does not match requested size.");
//    }
    
    std::vector<mxArray*> cells(input.size(),NULL);
    std::vector<std::string>::const_iterator it;
    for (it=input.begin(); it!=input.end(); it++) {
        mxArray* cell;
        writeString(*it, cell);
        cells[it-input.begin()] = cell;
    }
    
    writeCellArray(cells, output, m, n);
}

// template<typename T>
// void writeDoubleMatrix(const std::vector<T>& input, mxArray*& output, mwSize m, mwSize n)
// {
//     if (m*n != input.size()) {
//         mexErrMsgTxt("Input size does not match requested size.");
//     }
//     
//     output = mxCreateDoubleMatrix(m,n,mxREAL);
//     std::copy(input.begin(),input.end(),mxGetPr(output));
// }
// 
// template<typename T>
// void writeDoubleMatrix(const T* input, mxArray*& output, mwSize m, mwSize n)
// {
//     if (m*n != input.size()) {
//         mexErrMsgTxt("Input size does not match requested size.");
//     }
//     
//     output = mxCreateDoubleMatrix(m,n,mxREAL);
//     std::copy(input,input + m*n,mxGetPr(output));
// }

//void writeLogicalMatrix(const std::vector<bool>& input, mxArray*& output, mwSize m, mwSize n)
//{
//    if (m*n != input.size()) {
//        mexErrMsgTxt("Input size does not match requested size.");
//    }
    
//    output = mxCreateLogicalMatrix(m,n);
//    std::copy(input.begin(),input.end(),mxGetLogicals(output));
//}

// template<>
// void writeDoubleMatrix<double>(const std::vector<double>& input, mxArray*& output, mwSize m, mwSize n)
// {
//     if (m*n != input.size()) {
//         mexErrMsgTxt("Input size does not match requested size.");
//     }
//     
//     output = mxCreateDoubleMatrix(m,n,mxREAL);
//     std::copy(input.begin(),input.end(),mxGetPr(output));
// }
// 
// template<>
// void writeDoubleMatrix<float>(const std::vector<float>& input, mxArray*& output, mwSize m, mwSize n)
// {
//     if (m*n != input.size()) {
//         mexErrMsgTxt("Input size does not match requested size.");
//     }
//     
//     output = mxCreateDoubleMatrix(m,n,mxREAL);
//     std::copy(input.begin(),input.end(),mxGetPr(output));
// }
// 
// template<>
// void writeDoubleMatrix<unsigned int>(const std::vector<unsigned int>& input, mxArray*& output, mwSize m, mwSize n)
// {
//     if (m*n != input.size()) {
//         mexErrMsgTxt("Input size does not match requested size.");
//     }
//     
//     output = mxCreateDoubleMatrix(m,n,mxREAL);
//     std::copy(input.begin(),input.end(),mxGetPr(output));
// }
// 
// template<>
// void writeDoubleMatrix<mwIndex>(const std::vector<mwIndex>& input, mxArray*& output, mwSize m, mwSize n)
// {
//     if (m*n != input.size()) {
//         mexErrMsgTxt("Input size does not match requested size.");
//     }
//     
//     output = mxCreateDoubleMatrix(m,n,mxREAL);
//     std::copy(input.begin(),input.end(),mxGetPr(output));
// }
// 
// template<>
// void writeDoubleMatrix<int64_t>(const std::vector<int64_t>& input, mxArray*& output, mwSize m, mwSize n)
// {
//     if (m*n != input.size()) {
//         mexErrMsgTxt("Input size does not match requested size.");
//     }
//     
//     output = mxCreateDoubleMatrix(m,n,mxREAL);
//     std::copy(input.begin(),input.end(),mxGetPr(output));
// }
// 
// template<>
// void writeDoubleMatrix<uint64_t>(const std::vector<uint64_t>& input, mxArray*& output, mwSize m, mwSize n)
// {
//     if (m*n != input.size()) {
//         mexErrMsgTxt("Input size does not match requested size.");
//     }
//     
//     output = mxCreateDoubleMatrix(m,n,mxREAL);
//     std::copy(input.begin(),input.end(),mxGetPr(output));
// }

void writeCellArray(const std::vector<mxArray*>& input, mxArray*& output, mwSize m, mwSize n)
{
    if (m*n != input.size()) {
        mexErrMsgTxt("Input size does not match requested size.");
    }
    output = mxCreateCellMatrix(m,n);
    
    std::vector<mxArray*>::const_iterator it;
    for (it=input.begin(); it!=input.end(); it++) {
        mxSetCell(output,it-input.begin(),*it);
    }
}
