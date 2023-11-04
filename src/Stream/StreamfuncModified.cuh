#ifndef __STREAMFUNCMODIFIED_CUH
#define __STREAMFUNCMODIFIED_CUH
// in this version of the modified nonlinear and linear functions, we
// employ the colin's computation process totally. we introduce the intermediate
// variables H = \Delta(Q) - Q*(S^2-1). Due to the symmetry and asymmetry, we only
// need to consider the first and second elements h11 and h12.
// In addition, we evaluate the simplified version of the equation system on PRL
// this time
#include <Basic/QActFlow.h>
#include <Basic/Field.h>
#include <Basic/FldOp.cuh>

__global__ 
void vel_funcD(Qcomp* w_spec, Qcomp* u_spec, Qcomp* v_spec, 
                            Qreal* k_squared, Qreal* kx, Qreal*ky, int Nxh, int Ny, int BSZ);

void vel_func(Field *w, Field *u, Field *v);

__global__
void r1lin_func(Qreal* IFr1h, Qreal* IFr1, Qreal* k_squared, Qreal Pe, Qreal cn, Qreal dt, int Nxh, int Ny, int BSZ);

__global__
void r2lin_func(Qreal* IFr2h, Qreal* IFr2, Qreal* k_squared, Qreal Pe, Qreal cn, Qreal dt, int Nxh, int Ny, int BSZ);

__global__
void wlin_func(Qreal* IFwh, Qreal* IFw, Qreal* k_squared, Qreal Re, Qreal cf, Qreal dt, int Nxh, int Ny, int BSZ);

__global__ 
void S_funcD(Qreal* r1, Qreal*r2, Qreal* S, int Nx, int Ny, int BSZ);
void S_func(Field* r1, Field* r2, Field* S);

void curr_func(Field *r1curr, Field *r2curr, Field *wcurr, Field *u, Field *v, Field *S);

void r1nonl_func(Field *r1nonl, Field * u, Field * v, 
Field *S, Field *r1, Field *r2, Field *w, Field *h11, Qreal lambda, Field *aux);

void r2nonl_func(Field *r2nonl, Field * u, Field * v, 
Field *S, Field *r1, Field *r2, Field *w, Field *h12, Qreal lambda, Field *aux);

void wnonl_func(Field *wnonl, Field *h11,Field *h12, Field *p11, Field *p12, Field *p21, Field *r1, Field *r2, Field *w, 
                        Field *u, Field *v, Field *Ra, Field *S, Qreal Re, Qreal Er, Qreal lambda, Field *aux, Field *aux1);

// void pCross_func(Field *p,Field *aux, Field *r1, Field *r2);

// void pSingle_func(Field *p, Field *aux, Field *r, Field *S, Field *alpha, Qreal lambda, Qreal cn);

void h11nonl_func(Field *h11, Field *r1, Field *S);

void h12nonl_func(Field *h12, Field *r2, Field *S);

void p11nonl_func(Field *p11, Field *r1, Field *h11, Field *Ra, Field *S, Qreal lambda, Field *aux); 
void p12nonl_func(Field *p12, Field *r1, Field *r2, Field *h11, Field *h12, Field *Ra, Field *S, Qreal lambda, Field *aux); 
void p21nonl_func(Field *p21, Field *r1, Field *r2, Field *h11, Field *h12, Field *Ra, Field *S, Qreal lambda, Field *aux); 

#endif // end of __STREAMFUNCMODIFIED_CUH