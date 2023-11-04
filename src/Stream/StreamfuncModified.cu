#include <Stream/StreamfuncModified.cuh>

__global__ 
void vel_funcD(Qcomp* w_spec, Qcomp* u_spec, Qcomp* v_spec, 
                            Qreal* k_squared, Qreal* kx, Qreal*ky, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if (i==0 && j==0)
    {
        u_spec[index] = make_cuDoubleComplex(0.0,0.0);
        v_spec[index] = make_cuDoubleComplex(0.0,0.0);
    }
    else if(i<Nxh && j<Ny){
        //u = -D_y(\phi) -> u_spec = -1 * i* ky* w_spec/(-1* (kx^2+ky^2) )
        u_spec[index] = -1.0 * ky[j]*im()*w_spec[index]/(-1.0*k_squared[index]);
        //v = D_x(\phi) -> v_spec = i* kx* w_spec/(-1* (kx^2+ky^2) )
        v_spec[index] = kx[i]*im()*w_spec[index]/(-1.0*k_squared[index]);
    }
}
void vel_func(Field* w, Field* u, Field* v){
    int Nxh = w->mesh->Nxh;
    int Ny = w->mesh->Ny;
    int BSZ = w->mesh->BSZ;
    Qreal* k_squared = w->mesh->k_squared;
    Qreal* kx = w->mesh->kx;
    Qreal* ky = w->mesh->ky;
    dim3 dimGrid = w->mesh->dimGridsp;
    dim3 dimBlock = w->mesh->dimBlocksp; 
    vel_funcD<<<dimGrid, dimBlock>>>(w->spec, u->spec, v->spec, k_squared, kx, ky, Nxh, Ny, BSZ);
    BwdTrans(u->mesh, u->spec, u->phys);
    BwdTrans(v->mesh, v->spec, v->phys);
}

__global__ 
void S_funcD(Qreal* r1, Qreal* r2, Qreal* S, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        S[index] = 2*sqrt(r1[index]*r1[index] + r2[index]*r2[index]);
    }
}
void S_func(Field* r1, Field* r2, Field* S){
    Mesh* mesh = S->mesh;
    int Nx = mesh->Nx; int Ny = mesh->Ny; int BSZ = mesh->BSZ;
    dim3 dimGrid = mesh->dimGridp; dim3 dimBlock = mesh->dimBlockp;
    S_funcD<<<dimGrid, dimBlock>>>(r1->phys, r2->phys, S->phys, Nx, Ny, BSZ);
}

__global__
void xxDerivD(Qcomp *f_spec, Qcomp *fxx_spec, Qreal *kx, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        //D_xy(f) = -1*kx*kx*f_spec
        fxx_spec[index] = -1.0*f_spec[index]*kx[i]*kx[i];
    }
}
inline void xxDeriv(Field *f, Field* fxx){
    Mesh *mesh = f->mesh;
    int Nxh = mesh->Nxh; int Ny = mesh->Ny; int BSZ = mesh->BSZ;
    dim3 dimGrid = mesh->dimGridsp; dim3 dimBlock = mesh->dimBlocksp;
    xxDerivD<<<dimGrid, dimBlock>>>(f->spec, fxx->spec, mesh->kx, Nxh, Ny, BSZ);
    BwdTrans(mesh, fxx->spec, fxx->phys);
}

__global__
void yyDerivD(Qcomp *f_spec, Qcomp *fyy_spec, Qreal *ky, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        //D_xy(f) = -1*ky*ky*f_spec
        fyy_spec[index] = -1.0*f_spec[index]*ky[i]*ky[i];
    }
}
inline void yyDeriv(Field *f, Field* fyy){
    Mesh *mesh = f->mesh;
    int Nxh = mesh->Nxh; int Ny = mesh->Ny; int BSZ = mesh->BSZ;
    dim3 dimGrid = mesh->dimGridsp; dim3 dimBlock = mesh->dimBlocksp;
    yyDerivD<<<dimGrid, dimBlock>>>(f->spec, fyy->spec, mesh->ky, Nxh, Ny, BSZ);
    BwdTrans(mesh, fyy->spec, fyy->phys);
}

void xyDerivD(Qcomp *f_spec, Qcomp *fxy_spec, Qreal *kx, Qreal *ky, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        //D_xy(f) = -1*kx*ky*f_spec
        fxy_spec[index] = -1.0*f_spec[index]*kx[i]*ky[i];
    }
}
inline void xyDeriv(Field *f, Field* fxy){
    Mesh *mesh = f->mesh;
    int Nxh = mesh->Nxh; int Ny = mesh->Ny; int BSZ = mesh->BSZ;
    dim3 dimGrid = mesh->dimGridsp; dim3 dimBlock = mesh->dimBlocksp;
    xyDerivD<<<dimGrid, dimBlock>>>(f->spec, fxy->spec, mesh->kx, mesh->ky, Nxh, Ny, BSZ);
    BwdTrans(mesh, fxy->spec, fxy->phys);
}

inline void convect(Field *convf, Field *f, Field *u, Field *v, Field *aux){
    Mesh* mesh = convf->mesh;
    xDeriv(f->spec, convf->spec, mesh);
    BwdTrans(mesh, convf->spec, convf->phys);
    FldMul<<<mesh->dimGridp, mesh->dimBlockp>>>(convf->phys, u->phys, 1.0, convf->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

    yDeriv(f->spec, aux->spec, mesh);
    BwdTrans(mesh, aux->spec, aux->phys);
    FldMul<<<mesh->dimGridp, mesh->dimBlockp>>>(aux->phys, v->phys, 1.0, aux->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

    FldAdd<<<mesh->dimGridp, mesh->dimBlockp>>>(1.0, aux->phys, 1.0, convf->phys, convf->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    FwdTrans(mesh, convf->phys, convf->spec);
}
// this function designed to assist calculation of the multiple
// multiplication part of h calculation
// we assume that at this point the h (Dri) has stored the laplacian term
// of ri
__global__ 
void hmul_func(Qreal *Dri, Qreal *S, Qreal *r, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        // h = \Delta(r) - r*(S^2-1)
        Dri[index] = Dri[index] - r[index]*(S[index]*S[index] - 1);
    }
}
void hcal_func(Field *r, Field *S, Field *h){
    Mesh *mesh = h->mesh;
    laplacian_func(r->spec, h->spec, mesh);
    BwdTrans(mesh, h->spec, h->phys);
    hmul_func<<<mesh->dimGridp,mesh->dimBlockp>>>(h->phys, S->phys, r->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
}
void curr_func(Field *r1curr, Field *r2curr, Field *wcurr, Field *u, Field *v, 
Field *S, Field *h11, Field *h12){
    // obtain the physical values of velocities and r_i and h_{ij}
    // it is default that the accepted Field are all only update the spec
    
    int Nx = r1curr->mesh->Nx; int Ny = r1curr->mesh->Ny; int BSZ = r1curr->mesh->BSZ;
    dim3 dimGrid = r1curr->mesh->dimGridp; dim3 dimBlock = r1curr->mesh->dimBlockp;

    vel_func(wcurr, u, v);
    BwdTrans(r1curr->mesh,r1curr->spec, r1curr->phys);
    BwdTrans(r2curr->mesh,r2curr->spec, r2curr->phys);
    BwdTrans(wcurr->mesh, wcurr->spec, wcurr->phys);
    // calculate the physical val of S
    S_funcD<<<dimGrid, dimBlock>>>(r1curr->phys, r2curr->phys, S->phys, Nx, Ny, BSZ);
}


inline void h11nonl_func(Field *h11, Field *r1, Field *S){   
    hcal_func(r1, S, h11);
}

inline void h12nonl_func(Field *h12, Field *r2, Field *S){
    hcal_func(r2, S, h12);
}

__global__
void r1lin_func(Qreal* IFr1h, Qreal* IFr1, Qreal* k_squared, Qreal Pe, Qreal cn, Qreal dt, int Nxh, int Ny, int BSZ)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    double alpha1 = 1.0;
    if(i<Nxh && j<Ny){
        // IFr1h[index] = exp( alpha1*dt/2);
        IFr1h[index] = 1.0;
        // IFr1[index] = exp( alpha1*dt);
        IFr1[index] = 1.0;
    }
}

__global__
void r2lin_func(Qreal* IFr2h, Qreal* IFr2, Qreal* k_squared, Qreal Pe, Qreal cn, Qreal dt, int Nxh, int Ny, int BSZ)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    double alpha2 = 0.0;
    if(i<Nxh && j<Ny){
        // IFr2h[index] = exp( alpha2*dt/2);
        IFr2h[index] = 1.0;
        // IFr2[index] = exp( alpha2*dt);
        IFr2[index] = 1.0;
    }
}

__global__
void wlin_func(Qreal* IFwh, Qreal* IFw, Qreal* k_squared, Qreal Re, Qreal cf, Qreal dt, int Nxh, int Ny, int BSZ)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    //L(w) = 1/Re*(Laplacian(w) -cf*cf*w)
    // alpha0 = 1/Re*( -k_squared - cf*cf)
    Qreal alpha0 = 1.0/Re * (-1.0*k_squared[index] - cf*cf);
    if(i<Nxh && j<Ny){
        IFwh[index] = exp( alpha0 *dt/2);
        IFw[index] = exp( alpha0 *dt);
    }
}

void p11nonl_func(Field *p11, Field *r1, Field *h11, Field *Ra, Field *S, Qreal lambda, Field *aux){
    // p11 = -lambda*S*h11 -Ra*r1
    Mesh *mesh = p11->mesh;
    // aux = -lambda*S*h11
    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(S->phys, h11->phys, -1.0*lambda, aux->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

    // p11 = -Ra*r1
    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(Ra->phys, r1->phys, -1.0, p11->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

    // p11 = p11 + aux = -lambda*S*h11 -Ra*r1
    FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(1.0, aux->phys, 1.0, p11->phys, p11->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
}
void p12nonl_func(Field *p12, Field *r1, Field *r2, Field *h11, Field *h12, Field *Ra, Field *S, Qreal lambda, Field *aux){
    // p12 = 2*(r2h11 - r1h12) - lambda*Sh12 - Ra*r2
    Mesh *mesh = p12->mesh;
    // aux = 2*r2*h11
    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(r2->phys, h11->phys, 2.0, aux->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

    // p12 = -2*r1*h12
    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(r1->phys, h12->phys, -2.0, p12->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

    // p12 = p12 + aux = 2*(r2h11 - r1h12)
    FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(1.0, aux->phys, 1.0, p12->phys, p12->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

    // aux = -lambda * S * h12
    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(S->phys, h12->phys, -1.0*lambda, aux->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

    // p12 = p12 + aux = 2*(r2h11 - r1h12) -lambda * S * h12
    FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(1.0, aux->phys, 1.0, p12->phys, p12->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

    // aux = -Ra*r2
    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(Ra->phys, r2->phys, -1.0, aux->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

    // p12 = p12 + aux = 2*(r2h11 - r1h12) -lambda * S * h12 -Ra*r2
    FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(1.0, aux->phys, 1.0, p12->phys, p12->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

} 
void p21nonl_func(Field *p21, Field *r1, Field *r2, Field *h11, Field *h12, Field *Ra, Field *S, Qreal lambda, Field *aux){
    // p21 = 2*(r1h12 - r2h11) - lambda*Sh12 - Ra*r2
    Mesh *mesh = p21->mesh;
    // aux = -2*r2*h11
    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(r2->phys, h11->phys, -2.0, aux->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

    // p21 = 2*r1*h12
    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(r1->phys, h12->phys, 2.0, p21->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

    // p21 = p21 + aux = 2*(r1h12 - r2h11)
    FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(1.0, aux->phys, 1.0, p21->phys, p21->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

    // aux = -lambda * S * h12
    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(S->phys, h12->phys, -1.0*lambda, aux->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

    // p21 = p21 + aux = 2*(r2h11 - r1h12) -lambda * S * h12
    FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(1.0, aux->phys, 1.0, p21->phys, p21->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

    // aux = -Ra*r2
    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(Ra->phys, r2->phys, -1.0, aux->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

    // p21 = p21 + aux = 2*(r2h11 - r1h12) -lambda * S * h12 -Ra*r2
    FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(1.0, aux->phys, 1.0, p21->phys, p21->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
}

void r1nonl_func(Field *r1nonl, Field * u, Field * v, 
Field *S, Field *r1, Field *r2, Field *w, Field *h11, Qreal lambda, Field *aux){
    Mesh *mesh = r1nonl->mesh;
    // -1*convect(r1) + lambda*S*Dx(u) - r2*w + h11
    convect(r1nonl, r1, u, v, aux);

    // r1nonl = lambda*S*Dx(u)
    xDeriv(u->spec, r1nonl->spec, mesh);
    BwdTrans(mesh, r1nonl->spec, r1nonl->phys);
    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(r1nonl->phys, S->phys, lambda, r1nonl->phys, 
    mesh->Nx, mesh->Ny, mesh->BSZ);

    // r1nonl = r1nonl - convect(r1) = -1*convect(r1) + lambda*S*Dx(u)
    FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(1.0, r1nonl->phys, -1.0, aux->phys, r1nonl->phys, 
    mesh->Nx, mesh->Ny, mesh->BSZ);

    // aux = -r2*w 
    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(r2->phys, w->phys, -1.0, aux->phys, 
    mesh->Nx, mesh->Ny, mesh->BSZ);

    FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(1.0, r1nonl->phys, 1.0, aux->phys, r1nonl->phys, 
    mesh->Nx, mesh->Ny, mesh->BSZ);

    FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(1.0, r1nonl->phys, 1.0,h11->phys, r1nonl->phys, 
    mesh->Nx, mesh->Ny, mesh->BSZ);
}

void r2nonl_func(Field *r2nonl, Field * u, Field * v, 
Field *S, Field *r1, Field *r2, Field *w, Field *h12, Qreal lambda, Field *aux){
    Mesh *mesh = r2nonl->mesh;
    // -1*convect(r2) + 0.5*lambda*S*Dy(u) + 0.5*lambda*S*Dx(v) + r1*w + h12
    convect(r2nonl, r2, u, v, aux);

    // r1nonl = lambda*S*Dx(u)
    xDeriv(u->spec, r2nonl->spec, mesh);
    BwdTrans(mesh, r2nonl->spec, r2nonl->phys);
    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(r2nonl->phys, S->phys, lambda, r2nonl->phys, 
    mesh->Nx, mesh->Ny, mesh->BSZ);

    // r1nonl = r1nonl - convect(r1) = -1*convect(r1) + lambda*S*Dx(u)
    FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(1.0, r2nonl->phys, -1.0, aux->phys, r2nonl->phys, 
    mesh->Nx, mesh->Ny, mesh->BSZ);

    // aux = -r2*w 
    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(r2->phys, w->phys, -1.0, aux->phys, 
    mesh->Nx, mesh->Ny, mesh->BSZ);

    FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(1.0, r2nonl->phys, 1.0, aux->phys, r2nonl->phys, 
    mesh->Nx, mesh->Ny, mesh->BSZ);

    FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(1.0, r2nonl->phys, 1.0,h12->phys, r2nonl->phys, 
    mesh->Nx, mesh->Ny, mesh->BSZ);
}

void wnonl_func(Field *wnonl, Field *h11, Field *h12, Field *p11, Field *p12, Field *p21, Field *r1, Field *r2, Field *w, 
                        Field *u, Field *v, Field *Ra, Field *S, Qreal Re, Qreal Er, Qreal lambda, Field *aux, Field *aux1){
    Mesh* mesh = wnonl->mesh;
    p11nonl_func(p11, r1, h11, Ra, S, lambda, aux);
    p12nonl_func(p12, r1, r2, h11, h12, Ra, S, lambda, aux);
    p21nonl_func(p21, r1, r2, h11, h12, Ra, S, lambda, aux);

    xxDerivD<<<mesh->dimGridsp, mesh->dimBlocksp>>>(p12->spec, wnonl->spec, mesh->kx, mesh->Nxh, mesh->Ny, mesh->BSZ);
    xyDerivD<<<mesh->dimGridsp, mesh->dimBlocksp>>>(p11->spec, aux->spec, mesh->kx, mesh->ky, mesh->Nxh, mesh->Ny, mesh->BSZ);

    SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(1.0,aux->spec,-2.0,wnonl->spec,wnonl->spec,mesh->Nxh, mesh->Ny, mesh->BSZ);
    
    yyDerivD<<<mesh->dimGridsp, mesh->dimBlocksp>>>(p21->spec, aux->spec, mesh->kx, mesh->Nxh, mesh->Ny, mesh->BSZ);

    SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(1.0,wnonl->spec,-1.0,aux->spec,wnonl->spec,mesh->Nxh, mesh->Ny, mesh->BSZ);

    SpecMul(wnonl->spec, 1.0/(Re*Er), wnonl->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);

    convect(aux, w, u, v, aux1);

    SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(1.0,wnonl->spec,-1.0,aux->spec,wnonl->spec,mesh->Nxh, mesh->Ny, mesh->BSZ);
}
