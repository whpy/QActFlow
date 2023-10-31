#ifndef __QSOLVER_CUH
#define __QSOLVER_CUH

#include <Basic/QActFlow.h>
#include <Basic/FldOp.cuh>
#include <Basic/Field.h>
#include <Stream/Streamfunc.cuh>
#include <TimeIntegration/RK4.cuh>
#include <random>
#include <BasicUtils/UtilFuncs.hpp>

__global__ 
void r1_init(Qreal *r1, Qreal dx, Qreal dy, int Nx, int Ny, int BSZ);

__global__ 
void r2_init(Qreal *r2, Qreal dx, Qreal dy, int Nx, int Ny, int BSZ);

__global__ 
void w_init(Qreal *w, Qreal dx, Qreal dy, int Nx, int Ny, int BSZ);

inline 
void precompute_func(Field* r1, Field* r2, Field* w);
#endif