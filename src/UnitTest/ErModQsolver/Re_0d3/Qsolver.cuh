#ifndef __QSOLVER_CUH
#define __QSOLVER_CUH

#include <Basic/QActFlow.h>
#include <Basic/FldOp.cuh>
#include <Basic/Field.h>
#include <Stream/Streamfunc.cuh>
#include <TimeIntegration/RK4.cuh>

#include <stdlib.h>
#include <time.h>
#include <BasicUtils/UtilFuncs.hpp>

// to denote the type of initialization
enum InitType{
    Def_init,
    File_init
};
 
void r1_init(Qreal *r1, Qreal dx, Qreal dy, int Nx, int Ny);


void r2_init(Qreal *r2, Qreal dx, Qreal dy, int Nx, int Ny);


void w_init(Qreal *w, Qreal dx, Qreal dy, int Nx, int Ny);


void precompute_func(Field* r1, Field* r2, Field* w, InitType flag);

#endif