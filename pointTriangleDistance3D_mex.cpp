#include "mex.h"

#include<vector>

#include "mexutils.h"
#include "geom.h"

// [d,b1,b2,b3] = pointTriangleDistance3D_mex(p,tv1,tv2,tv3)
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if(nrhs != 4) {
        mexErrMsgTxt("Four inputs required.");
    }
    if(nlhs < 1 || nlhs > 4) {
        mexErrMsgTxt("One to four outputs required.");
    }
    
    mwSize dim ,n, np, nt;
    
    // read points
    std::vector<double> p;
    readDoubleMatrix(prhs[0], p, dim, np);
    if (dim != 3) {
        mexErrMsgTxt("Input must be three-dimensional.");
    }
    
    // read triangles
    std::vector<double> t1;
    std::vector<double> t2;
    std::vector<double> t3;
    readDoubleMatrix(prhs[1], t1, dim, nt);
    if (dim != 3) {
        mexErrMsgTxt("Input must be three-dimensional.");
    }
    
    readDoubleMatrix(prhs[2], t2, dim, n);
    if (dim != 3) {
        mexErrMsgTxt("Input must be three-dimensional.");
    }
    if (n != nt) {
        mexErrMsgTxt("Input size does not match.");
    }
    
    readDoubleMatrix(prhs[3], t3, dim, n);
    if (dim != 3) {
        mexErrMsgTxt("Input must be three-dimensional.");
    }
    if (n != nt) {
        mexErrMsgTxt("Input size does not match.");
    }
    
    // compute distance
    std::vector<double> dist(np*nt);
    std::vector<double> b1(np*nt);
    std::vector<double> b2(np*nt);
    std::vector<double> b3(np*nt);
    if (np >= 1 && nt >= 1) {
        mwIndex ind;
        Vector point;
        Triangle tri;
        for (mwIndex ti=0; ti<nt; ti++) {
            for (mwIndex pi=0; pi<np; pi++) {
                
                ind = pi*dim;
                point = Vector(p[ind],p[ind+1],p[ind+2]);

                ind = ti*dim;
                tri.P0 = Vector(t1[ind],t1[ind+1],t1[ind+2]);
                tri.P1 = Vector(t2[ind],t2[ind+1],t2[ind+2]);
                tri.P2 = Vector(t3[ind],t3[ind+1],t3[ind+2]);
                
                ind = ti*np+pi;
                dist[ind] = pointTriangleDistance3D(point,tri,b1[ind],b2[ind],b3[ind]);
            }
        }
    }
    p.clear();
    t1.clear();
    t2.clear();
    t3.clear();
    
    // write outputs
    plhs[0] = mxCreateDoubleMatrix(np,nt,mxREAL);
    std::copy(dist.begin(),dist.end(),mxGetPr(plhs[0]));
    dist.clear();
    
    if (nlhs >= 2) {
        plhs[1] = mxCreateDoubleMatrix(np,nt,mxREAL);
        std::copy(b1.begin(),b1.end(),mxGetPr(plhs[1]));
    }
    b1.clear();
    
    if (nlhs >= 3) {
        plhs[2] = mxCreateDoubleMatrix(np,nt,mxREAL);
        std::copy(b2.begin(),b2.end(),mxGetPr(plhs[2]));
    }
    b2.clear();
    
    if (nlhs >= 4) {
        plhs[3] = mxCreateDoubleMatrix(np,nt,mxREAL);
        std::copy(b3.begin(),b3.end(),mxGetPr(plhs[3]));
    }
    b3.clear();
}
