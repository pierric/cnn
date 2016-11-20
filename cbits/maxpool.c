#include <stdlib.h>
#include <string.h>

/*  max pooling of stride 2.
    assuming that
    - row and col are both multiple of 2.
    - mat is of row-major.
 */
void pool2_f(int row, int col, int stride, float *mat,
             float *max, int *ind)
{
    int rc = row / 2;
    int cc = col / 2;
    float *p1 = mat;
    float *p2 = p1+stride;
    for (int i=0;i<rc;i++) {
        float *r1 = p1;
        float *r2 = p2;
        for (int j=0;j<cc;j++) {
            *max = r1[0];
            *ind = 0;
            if (r1[1] > *max) { *max = r1[1]; *ind = 1; }
            if (r2[0] > *max) { *max = r2[0]; *ind = 2; }
            if (r2[1] > *max) { *max = r2[1]; *ind = 3; }
            max++;
            ind++;
            r1+=2;
            r2+=2;
        }
        p1+=2*stride;
        p2+=2*stride;
    }
}
