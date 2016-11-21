#include <stdlib.h>
#include <string.h>
#include "cblas.h"

#define AT(p,s,x,y) ((p)+(s)*(x)+(y))


// 2M byte working storage per thread.
// sufficient to handle correlation/convolution wthin
// the size (5x5, 128x128) or (7x7, 100x100)
#define WORKINGSTORAGE 524288
static _Thread_local float workingSto[WORKINGSTORAGE];

/*
input:  two matrices, assuming in row major, may not be in continuous. i.e.
        one row has only column number of elements, but its length is of
        the stride.
        mat1 is the kernel
        mat2 is the source
        assuming that row1<=row2 and col1 <= col2
output: continuous row-major matrix of size u x v
        where u = row2-row1+1
              v = col2-co11+1
*/

int corr_sf_general(
    int reversemat1, int row1, int col1, int stride1, float *mat1,
    int row2, int col2, int stride2, float *mat2, float *mat3)
{
    int u = row2-row1+1;
    int v = col2-col1+1;
    if ((u*v+1)*row1*col1 > WORKINGSTORAGE) return -1;
    float *ws = workingSto;
    for(int i=0;i<u;i++) {
        for(int j=0;j<v;j++) {
            for(int k=0;k<row1;k++) {
                float *src=AT(mat2,stride2,i+k,j);
                memcpy(ws, src, col1*sizeof(float));
                ws += col1;
            }
        }
    }
    float *vw = mat1;
    if (stride1 > col1 || reversemat1) {
        // we have expecting an extra row1*col1 elements at hand, pointed
        // by ws at this point of time, and used for a continues
        // copy of the kernel matrix.
        vw = ws;
        float *p1=vw, *p2=mat1;
        for(int i=0;i<row1;i++) {
            memcpy(p1,p2,col1*sizeof(float));
            p1 += col1;
            p2 += stride1;
        }
    }
    // reverse the kernel when do convolution.
    if(reversemat1) {
        int sz = row1*col1;
        for(int i=0;i<sz/2;i++) {
            float t = vw[i];
            vw[i] = vw[sz-1-i];
            vw[sz-1-i] = t;
        }
    }
    // execute the matrix vector multiplication
    ws = workingSto;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, u*v, row1*col1,
    		    1.0, ws, row1*col1, vw, 1, 0, mat3, 1);
    return 0;
}
