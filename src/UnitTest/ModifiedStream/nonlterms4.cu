#include <Basic/QActFlow.h>
#include <Basic/FldOp.cuh>
#include <Basic/Field.h>
#include <Stream/StreamfuncModified.cuh>
#include <TimeIntegration/RK4.cuh>
#include <iostream>

using namespace std;

// \phi = sin(2*x+3*y)
// r1 = cos(3*x+2*y)
// r2 = sin(3*x+2*y)

__global__
void PhiinitD(Qreal* phys, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    if (i<Nx && j<Ny){
        phys[index] = sin(2*x + 3*y);
    }
}

__global__
void rinitD(Qreal* r1, Qreal* r2, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    if (i<Nx && j<Ny){
        r1[index] = cos(3*x + 2*y);
        r2[index] = sin(3*x + 2*y);
    }
}
__global__
void RainitD(Qreal* phys, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    if (i<Nx && j<Ny){
        phys[index] = 0.2;
    }
}


// S = 2*sqrt(r1^2+r2^2) = 2
__global__
void SexactD(Qreal* S, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    if (i<Nx && j<Ny){
        S[index] = 2.0;
    }
}

__global__
void wexactD(Qreal* phys, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    if (i<Nx && j<Ny){
        phys[index] = -5.0*sin(2*x+y);
    }
}



__global__
void uexactD(Qreal* phys, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    // Qreal s = exp(-1*( (x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI) ));
    if (i<Nx && j<Ny){
        phys[index] = -1*cos(2*x+y);
    }
}
__global__
void vexactD(Qreal* phys, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    // Qreal s = exp(-1*( (x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI) ));
    if (i<Nx && j<Ny){
        phys[index] = 2*cos(2*x+y);
    }
}

__global__
void NL1exactD(Qreal* phys, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    // Qreal s = exp(-1*( (x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI) ));
    if (i<Nx && j<Ny){
        phys[index] = -16*cos(3*x+2*y) - 5*cos(2*x+3*y)*sin(3*x+2*y) + 
        13.0*(0.0923077 + sin(3*x+2*y)) * sin(2*x+3*y);
    }
}

__global__
void NL2exactD(Qreal* phys, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    // Qreal s = exp(-1*( (x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI) ));
    if (i<Nx && j<Ny){
        phys[index] = -16*sin(3*x+2*y) + cos(3*x+2*y)*(5*cos(2*x+3*y) - 13*sin(2*x+3*y))
        + 0.5*sin(2*x+3*y);
    }
}

__global__
void NL0exactD(Qreal* phys, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    if (i<Nx && j<Ny){
        phys[index] = 3600*cos(3*x+2*y) - 1500*sin(3*x+2*y);
    }
}

__global__
void p11exact(Qreal* phys, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    Qreal s = exp(-1*( (x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI) ));
    if (i<Nx && j<Ny){
        phys[index] = 1.4*cos(2*x+y);
    }
}

__global__
void p12exact(Qreal* phys, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    Qreal s = exp(-1*( (x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI) ));
    if (i<Nx && j<Ny){
        phys[index] = 1.4*sin(2*x+y);
    }
}
__global__
void p21exact(Qreal* phys, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    Qreal s = exp(-1*( (x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI) ));
    if (i<Nx && j<Ny){
        phys[index] = 1.4*sin(2*x+y);
    }
}

__global__
void h11exactD(Qreal* phys, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    Qreal s = exp(-1*( (x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI) ));
    if (i<Nx && j<Ny){
        phys[index] = 1.4*cos(2*x+y);
    }
}

__global__
void h12exactD(Qreal* phys, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    Qreal s = exp(-1*( (x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI) ));
    if (i<Nx && j<Ny){
        phys[index] = -5.0*sin(x+y);
    }
}


void field_visual(Field *f, string name){
    Mesh* mesh = f->mesh;
    ofstream fval;
    string fname = name;
    fval.open(fname);
    for (int j=0; j<mesh->Ny; j++){
        for (int i=0; i<mesh->Nx; i++){
            fval << f->phys[j*mesh->Nx+i] << ",";
        }
        fval << endl;
    }
    fval.close();
}

inline void init(Field *Ra, Field* Phi, Field *r1, Field *r2,Field* we, Field* ue, Field* ve, Field* h11e, Field* h12e, 
Field* wnonle, Field* r1nonle, Field* r2nonle){
    Mesh* mesh = we->mesh;
    int Nx = mesh->Nx;
    int Ny = mesh->Ny;
    Qreal dx = mesh->dx;
    Qreal dy = mesh->dy;
    int BSZ = mesh->BSZ;
    dim3 dimGrid = mesh->dimGridp;
    dim3 dimBlock = mesh->dimBlockp;
    PhiinitD<<<dimGrid, dimBlock>>>(Phi->phys, Nx, Ny, BSZ, dx,dy);
    RainitD<<<dimGrid, dimBlock>>>(Ra->phys, Nx,Ny, BSZ, dx,dy);
    rinitD<<<dimGrid, dimBlock>>>(r1->phys, r2->phys, Nx, Ny, BSZ, dx, dy);
    wexactD<<<dimGrid, dimBlock>>>(we->phys, Nx, Ny, BSZ,dx, dy);
    uexactD<<<dimGrid, dimBlock>>>(ue->phys, Nx, Ny, BSZ, dx, dy);
    vexactD<<<dimGrid, dimBlock>>>(ve->phys, Nx, Ny, BSZ, dx, dy);
    h11exactD<<<dimGrid, dimBlock>>>(h11e->phys, Nx, Ny, BSZ, dx, dy);
    h12exactD<<<dimGrid, dimBlock>>>(h12e->phys, Nx, Ny, BSZ, dx, dy);
    NL0exactD<<<dimGrid, dimBlock>>>(wnonle->phys, Nx, Ny, BSZ, dx, dy);
    NL1exactD<<<dimGrid, dimBlock>>>(r1nonle->phys, Nx, Ny, BSZ, dx, dy);
    NL2exactD<<<dimGrid, dimBlock>>>(r2nonle->phys, Nx, Ny, BSZ, dx, dy);

    FwdTrans(mesh, r1->phys, r1->spec);
    FwdTrans(mesh, r2->phys, r2->spec);
    cuda_error_func(cudaDeviceSynchronize());
    field_visual(r1,"r1.csv");
    field_visual(r2,"r2.csv");
    field_visual(Ra,"Ra.csv");
    field_visual(we,"wexact.csv");
    field_visual(ue,"uexact.csv");
    field_visual(ve,"vexact.csv");
    field_visual(h11e,"h11exact.csv");
    field_visual(h12e,"h12exact.csv");
    field_visual(wnonle,"wnlexact.csv");
    field_visual(r1nonle,"r1nlexact.csv");
    field_visual(r2nonle,"r2nlexact.csv");

}

void print_spec(Field* f){
    Mesh* mesh = f->mesh;
    int Nxh = mesh->Nxh, Ny = mesh->Ny;
    for(int j = 0; j < Ny; j++){
        for (int i = 0; i < Nxh; i++){
            int index = i + j*Nxh;
            cout << "("<< f->spec[index].x << "," << f->spec[index].y << ")" << " ";
        }
        cout << endl;
    }
}

void print_phys(Field* f){
    Mesh* mesh = f->mesh;
    int Nx = mesh->Nx, Ny = mesh->Ny;
    for(int j = 0; j < Ny; j++){
        for (int i = 0; i < Nx; i++){
            int index = i + j*Nx;
            cout << "("<< f->phys[index]<< ")" << " ";
        }
        cout << endl;
    }
}

void coord(Mesh &mesh){
    ofstream xcoord("x.csv");
    ofstream ycoord("y.csv");
    for (int j=0; j<mesh.Ny; j++){
        for ( int i=0; i< mesh.Nx; i++){
            Qreal x = mesh.dx*i;
            Qreal y = mesh.dy*j;
            xcoord << x << ",";
            ycoord << y << ",";
        }
        xcoord << endl;
        ycoord << endl;
    }
    xcoord.close();
    ycoord.close();
}

// we test the single func and the cross term func in
// this module, with the velocity computation. 
// the r1 is set to be 1/2*sin(x+y)*x/2pi and the r2 is set to be 1/2*cos(x+y)*y/2pi
// which satisfies that r1^2+r2^2<=0.25. S is derived to be 1/(16pi)*(x^2+y^2);
// lap(r1) = 1/(2*pi)*(cos(x+y) - x*sin(x+y)), 
// lap(r2) = 1/(2*pi)*(-1)*(sin(x+y) + y*cos(x+y))
// the cross term should be 2*(r2*Lap(r1) - r1*Lap(r2)) = 
int main(){
     // computation parameters
    int BSZ = 16;
    int Ns = 20000;
    int Nx = 512; // same as colin
    int Ny = 512;
    int Nxh = Nx/2+1;
    Qreal Lx =  4 * 2 *M_PI;
    Qreal Ly = Lx;
    Qreal dx = Lx/Nx;
    Qreal dy = dx;
    Qreal dt = 0.001; // same as colin

    // non-dimensional number
    Qreal Re = 0.1;
    Qreal Er = 0.1;
    Qreal Rf = 7.5 * 0.00001;
    Qreal lambda = 0.1;

    // main Fields to be solved
    // *_curr act as an intermediate while RK4 timeintegration
    // *_new store the value of next time step 

    Mesh *mesh = new Mesh(BSZ, Nx, Ny, Lx, Ly);
    cout << "Re = " << Re << endl;
    cout << "Er = " << Er << endl;
    cout << "lambda = " << lambda << endl;
    cout << "Rf = " << Rf << endl;
    cout<< "Lx: " << mesh->Lx << " "<< "Ly: " << mesh->Ly << " " << endl;
    cout<< "Nx: " << mesh->Nx << " "<< "Ny: " << mesh->Ny << " " << endl;
    cout<< "dx: " << mesh->dx << " "<< "dy: " << mesh->dy << " " << endl;
    cout<< "Nx*dx: " << mesh->Nx*mesh->dx << " "<< "Ny*dy: " << mesh->Ny*mesh->dy << " " << endl;
    Field *w_old = new Field(mesh); Field *w_curr = new Field(mesh); Field *w_new = new Field(mesh);
    Field *r1_old = new Field(mesh); Field *r1_curr = new Field(mesh); Field *r1_new = new Field(mesh);
    Field *r2_old = new Field(mesh); Field *r2_curr = new Field(mesh); Field *r2_new = new Field(mesh);

     // linear factors
    Qreal *wIF, *wIFh; Qreal *r1IF, *r1IFh; Qreal *r2IF, *r2IFh;
    cudaMallocManaged(&wIF, sizeof(Qreal)*Nxh*Ny); cudaMallocManaged(&wIFh, sizeof(Qreal)*Nxh*Ny);
    cudaMallocManaged(&r1IF, sizeof(Qreal)*Nxh*Ny); cudaMallocManaged(&r1IFh, sizeof(Qreal)*Nxh*Ny);
    cudaMallocManaged(&r2IF, sizeof(Qreal)*Nxh*Ny); cudaMallocManaged(&r2IFh, sizeof(Qreal)*Nxh*Ny);

    // intermediate fields
    //nonlinear terms
    Field *wnonl = new Field(mesh); Field *r1nonl = new Field(mesh); Field *r2nonl = new Field(mesh);
    // velocity and S
    Field *u = new Field(mesh); Field *v = new Field(mesh); Field *S = new Field(mesh);
    // H tensor
    Field *h11 = new Field(mesh); Field *h12 = new Field(mesh);
    // the stress tensor
    Field *p11 = new Field(mesh); Field *p12 = new Field(mesh); Field* p21 = new Field(mesh);
    // auxiliary fields
    Field *aux = new Field(mesh); Field *aux1 = new Field(mesh); 

    // field \alpha to be modified (scalar at the very first)
    Field *Ra = new Field(mesh);
// /////////// 2 ////////////
// // we previously have verified the validity of laplacian and vel_func.
// // in this file we test the func about the Q tensor (components, r1, r2) and 
// // the intermediate components p (p11, p12, p21)
//     Mesh *mesh = new Mesh(BSZ, Nx, Ny, Lx, Ly);
//     Field* phi = new Field(mesh);
//     Field* w = new Field(mesh); Field* wa = new Field(mesh);
//     Field* u = new Field(mesh); Field* ua = new Field(mesh);
//     Field* v = new Field(mesh); Field* va = new Field(mesh);

//     Field *r1 = new Field(mesh); Field *r2 = new Field(mesh);
//     Field *S = new Field(mesh); Field *Sa = new Field(mesh);
    
//     Field *single1 = new Field(mesh); Field *single1a = new Field(mesh);
//     Field *single2 = new Field(mesh); Field *single2a = new Field(mesh);
//     Field *cross1 = new Field(mesh); Field *cross2 = new Field(mesh); 
//     Field *crossa = new Field(mesh);

//     Field *p11 = new Field(mesh); Field *p11a = new Field(mesh);
//     Field *p12 = new Field(mesh); Field *p12a = new Field(mesh);
//     Field *p21 = new Field(mesh); Field *p21a = new Field(mesh);

//     Field *nl0 = new Field(mesh); Field *nl0a = new Field(mesh);
//     Field *nl1 = new Field(mesh); Field *nl1a = new Field(mesh);
//     Field *nl2 = new Field(mesh); Field *nl2a = new Field(mesh);
//     Field *alpha = new Field(mesh);

    // aux is the abbreviation of auxiliary, where only act as intermediate values
    // to assist computation. So we should guarantee that it doesnt undertake any 
    // long term memory work.
    // Field *aux = new Field(mesh); Field *aux1 = new Field(mesh);
    // Field* phi = new Field(mesh);
    // Field* w = new Field(mesh); Field* wa = new Field(mesh);
    // Field* u = new Field(mesh); Field* ua = new Field(mesh);
    // Field* v = new Field(mesh); Field* va = new Field(mesh);

    Field *we = new Field(mesh); Field *ue = new Field(mesh); Field *ve = new Field(mesh); 
    Field *h11e = new Field(mesh); Field *h12e = new Field(mesh); Field *wnonle = new Field(mesh); 
    Field *p11e = new Field(mesh); Field *p12e = new Field(mesh); Field *p21e = new Field(mesh);
    Field *r1nonle = new Field(mesh); Field *r2nonle = new Field(mesh);
    Field *phi = new Field(mesh);
    coord(*mesh);
    init(Ra, phi, r1_old, r2_old, we,ue,ve, h11e,h12e,wnonle,r1nonle,r2nonle);
//     FldSet<<<mesh->dimGridp, mesh->dimBlockp>>>(alpha->phys, 1.0, Nx, Ny, BSZ);
    p11exact<<<mesh->dimGridp,mesh->dimBlockp>>>(p11e->phys, Nx, Ny, BSZ, dx, dy);
    p12exact<<<mesh->dimGridp,mesh->dimBlockp>>>(p12e->phys, Nx, Ny, BSZ, dx, dy);
    p21exact<<<mesh->dimGridp,mesh->dimBlockp>>>(p21e->phys, Nx, Ny, BSZ, dx, dy);
    // evaluate the spectral of w: Laplacian( Four(\phi) )
    FwdTrans(mesh,phi->phys, phi->spec);
    laplacian_func(phi->spec, w_old->spec, mesh);
    // switch back
    BwdTrans(mesh, w_old->spec, w_old->phys);
    
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(w_old, "w.csv");
    field_visual(p11e, "p11e.csv");
    field_visual(p12e, "p12e.csv");
    field_visual(p21e, "p21e.csv");
    field_visual(wnonl, "wnonl_start.csv");
    field_visual(r1nonl, "r1nonl_start.csv");
    field_visual(r2nonl, "r2nonl_start.csv");
    
//     // evaluate the S; the velocity u, v; the phys of r1, r2; 

// //////////////////// Sfunc tested //////////////////

// //////////////////// crossfunc test ///////////////

// //////////////////// cross and single term tested //////////////////

// //////////////////// nonlinear term test //////////////////

    //////////////////////// precomputation //////////////////////////
    r1lin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1IFh, r1IF, dt, r1_old->mesh->Nxh, r1_old->mesh->Ny, r1_old->mesh->BSZ);
    r2lin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2IFh, r2IF, dt, r2_old->mesh->Nxh, r2_old->mesh->Ny, r2_old->mesh->BSZ);
    wlin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wIFh, wIF, w_old->mesh->k_squared, Re, Rf, dt, w_old->mesh->Nxh, w_old->mesh->Ny, w_old->mesh->BSZ);
    integrate_func0(w_old, w_curr, w_new, wIF, wIFh, dt);
    integrate_func0(r1_old, r1_curr, r1_new, r1IF, r1IFh, dt);
    integrate_func0(r2_old, r2_curr, r2_new, r2IF, r2IFh, dt);

    curr_func(r1_curr, r2_curr,w_curr,u,v,S, h11, h12);
    convect(aux, w_curr, u, v, aux1);
    BwdTrans(mesh, w_curr->spec, w_curr->phys);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(w_curr, "I0w.csv");
    field_visual(aux,"convectw.csv");
    field_visual(h11, "I0h11.csv");
    field_visual(h12, "I0h12.csv");
    field_visual(u, "I0u.csv");
    field_visual(v, "I0v.csv");
    field_visual(S, "I0S.csv");
    field_visual(r1_curr, "I0r1.csv");
    field_visual(r2_curr, "I0r2.csv");
    r1nonl_func(r1nonl, u, v, S, r1_curr, r2_curr, w_curr, h11, lambda, aux, aux1);
    r2nonl_func(r2nonl, u, v, S, r1_curr, r2_curr, w_curr, h12, lambda, aux, aux1); 
    wnonl_func(wnonl, h11,h12,p11,p12,p21,r1_curr,r2_curr,w_curr,u,v,Ra,S,Re, Er, lambda,aux,aux1);
    /************************************/
    // p11nonl_func(p11, r1_curr, h11, Ra, S, lambda, aux);
    // p12nonl_func(p12, r1_curr, r2_curr, h11, h12, Ra, S, lambda, aux);
    // p21nonl_func(p21, r1_curr, r2_curr, h11, h12, Ra, S, lambda, aux);
    
    // //Dxx(p12)
    // // xxDerivD<<<mesh->dimGridsp, mesh->dimBlocksp>>>(p12->spec, wnonl->spec, mesh->kx, mesh->Nxh, mesh->Ny, mesh->BSZ);
    // xxDeriv(p12, aux);
    // cuda_error_func(cudaDeviceSynchronize());
    // field_visual(aux, "Dxx(p12).csv");
    // //Dxy(p11)
    // // xyDerivD<<<mesh->dimGridsp, mesh->dimBlocksp>>>(p11->spec, aux->spec, mesh->kx, mesh->ky, mesh->Nxh, mesh->Ny, mesh->BSZ);
    // xyDeriv(p11, wnonl);
    // cuda_error_func(cudaDeviceSynchronize());
    // field_visual(wnonl, "Dxy(p11).csv");
    // // wnonl.spec = Dxx(p12) - 2*Dxy(p11)
    // FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(1.0, aux->phys, -2.0, wnonl->phys, wnonl->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    // // SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(1.0,aux->spec,-2.0,wnonl->spec,wnonl->spec,mesh->Nxh, mesh->Ny, mesh->BSZ);
    // cuda_error_func(cudaDeviceSynchronize());
    // field_visual(wnonl, "first.csv");
    // //Dyy(p21)
    // // yyDerivD<<<mesh->dimGridsp, mesh->dimBlocksp>>>(p21->spec, aux->spec, mesh->kx, mesh->Nxh, mesh->Ny, mesh->BSZ);
    // // wnonl.spec = Dxx(p12) - 2*Dxy(p11) - Dyy(p21)
    // // SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(1.0,wnonl->spec,-1.0,aux->spec,wnonl->spec,mesh->Nxh, mesh->Ny, mesh->BSZ);
    // yyDeriv(p21, aux);
    // // FwdTrans(mesh,p21e->phys, p21e->spec);
    // // yyDeriv(p21e, aux1);
    // cuda_error_func(cudaDeviceSynchronize());
    // field_visual(aux, "Dyy(p21).csv");
    // // field_visual(aux1, "Dyy(p21)e.csv");
    // FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(1.0, wnonl->phys, -1.0, aux->phys, wnonl->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    // cuda_error_func( cudaDeviceSynchronize() );
    // field_visual(wnonl, "second.csv");
    // // wnonl.spec = 1.0/(Re*Er)(Dxx(p12) - 2*Dxy(p11) - Dyy(p21))
    // // SpecMul<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wnonl->spec, 1.0/(Re*Er), wnonl->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
    // // aux->spec = uDx(w)+vDy(w);
    // // FwdTrans(mesh, wnonl->phys, wnonl->spec);
    // convect(aux, w_curr, u, v, aux1);
    // // wnonl.spec = 1.0/(Re*Er)(Dxx(p12) - 2*Dxy(p11) - Dyy(p21)) - (uDx(w)+vDy(w))
    // // SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(1.0/(Re*Er),wnonl->spec,-1.0,aux->spec,wnonl->spec,mesh->Nxh, mesh->Ny, mesh->BSZ);
    // FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(1.0/(Re*Er),wnonl->phys, -1.0, aux->phys, wnonl->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    // cuda_error_func( cudaDeviceSynchronize() );
    // field_visual(wnonl, "third.csv");
    // FwdTrans(mesh, wnonl->phys, wnonl->spec);
    /***********************************/ 

    BwdTrans(mesh,wnonl->spec,wnonl->phys);
    BwdTrans(mesh, p11->spec, p11->phys);
    BwdTrans(mesh, p12->spec, p12->phys);
    BwdTrans(mesh, p21->spec, p21->phys);
    // FldAdd<<<mesh->dimGridp, mesh->dimBlockp>>>(1.0, p12->phys, -1.0, p21->phys, aux->phys, Nx, Ny, BSZ);
    FwdTrans(mesh, p21->phys, p21->spec);
    FwdTrans(mesh, p12->phys, p21->spec);
    // xxDeriv(p12, aux);
    // yyDeriv(p21, aux1);
    // FldAdd<<<mesh->dimGridp, mesh->dimBlockp>>>(1.0, aux->phys, -1.0, aux1->phys, aux->phys, Nx, Ny, BSZ);
    // SpecAdd<<<mesh->dimGridsp, mesh->dimGridsp>>>(1.0, aux->spec, -1.0, aux1->spec, aux->spec, Nxh ,Ny, BSZ);
    // xyDeriv(p11, aux1);
    // FldAdd<<<mesh->dimGridp, mesh->dimBlockp>>>(1.0, aux->phys, -2.0, aux1->phys, aux->phys, Nx, Ny, BSZ);
    // SpecAdd<<<mesh->dimGridsp, mesh->dimGridsp>>>(-2.0, aux1->spec, 1.0, aux->spec, aux->spec, Nxh ,Ny, BSZ);
    // BwdTrans(mesh, aux->spec, aux->phys);
    FwdTrans(mesh, p11e->phys, p11e->spec);
    xxDeriv(p11e, aux);

    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(aux, "xxD.csv");
    field_visual(wnonl, "I0wnonl.csv");
    field_visual(r1nonl, "I0r1nonl.csv");
    field_visual(r2nonl, "I0r2nonl.csv");
    field_visual(p11, "p11.csv");
    field_visual(p12, "p12.csv");
    field_visual(p21, "p21.csv");
    return 0;
}