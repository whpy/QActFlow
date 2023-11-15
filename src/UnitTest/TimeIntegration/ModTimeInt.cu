#include <Basic/QActFlow.h>
#include <Basic/FldOp.cuh>
#include <Basic/Field.h>
#include <Stream/StreamfuncModified.cuh>
#include <TimeIntegration/RK4.cuh>
#include <iostream>

using namespace std;

// \phi = cos(3x)*sin(4y)
// w = Laplacian(\phi) = -25*cos(3x)*sin(4y)
// u = -1*Dy(\phi) = -4*cos(3x)*cos(4y)
// v = Dx(\phi) = -3*sin(3x)*sin(4y)

__global__
void PhiinitD(Qreal* phys, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    if (i<Nx && j<Ny){
        phys[index] = sin(x + y);
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
__global__
void rinitD(Qreal* r1, Qreal* r2, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    if (i<Nx && j<Ny){
        r1[index] = cos(x + y);
        r2[index] = sin(x + y);
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

inline void init(Field *Ra, Field* Phi, Field *r1, Field *r2){
    Mesh* mesh = r1->mesh;
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

    FwdTrans(mesh, r1->phys, r1->spec);
    FwdTrans(mesh, r2->phys, r2->spec);
    cuda_error_func(cudaDeviceSynchronize());
    field_visual(r1,"r1.csv");
    field_visual(r2,"r2.csv");
    field_visual(Ra,"Ra.csv");
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

    Field *phi = new Field(mesh);
    coord(*mesh);
    init(Ra, phi, r1_old, r2_old);
//     FldSet<<<mesh->dimGridp, mesh->dimBlockp>>>(alpha->phys, 1.0, Nx, Ny, BSZ);
    // evaluate the spectral of w: Laplacian( Four(\phi) )
    FwdTrans(mesh,phi->phys, phi->spec);
    laplacian_func(phi->spec, w_old->spec, mesh);
    // switch back
    BwdTrans(mesh, w_old->spec, w_old->phys);
    
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(w_old, "w.csv");
    field_visual(wnonl, "wnonl_start.csv");
    field_visual(r1nonl, "r1nonl_start.csv");
    field_visual(r2nonl, "r2nonl_start.csv");
    
    // evaluate the S; the velocity u, v; the phys of r1, r2; 

    //////////////////////// precomputation //////////////////////////
    r1lin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1IFh, r1IF, dt, r1_old->mesh->Nxh, r1_old->mesh->Ny, r1_old->mesh->BSZ);
    r2lin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2IFh, r2IF, dt, r2_old->mesh->Nxh, r2_old->mesh->Ny, r2_old->mesh->BSZ);
    wlin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wIFh, wIF, w_old->mesh->k_squared, Re, Rf, dt, w_old->mesh->Nxh, w_old->mesh->Ny, w_old->mesh->BSZ);
    
    ////////////////////// Time Int tested ////////////////////////
    ///////////////////// Integrate 0 ////////////////////////////
    integrate_func0(w_old, w_curr, w_new, wIF, wIFh, dt);
    integrate_func0(r1_old, r1_curr, r1_new, r1IF, r1IFh, dt);
    integrate_func0(r2_old, r2_curr, r2_new, r2IF, r2IFh, dt);

    curr_func(r1_curr, r2_curr,w_curr,u,v,S, h11, h12);
    BwdTrans(mesh, w_curr->spec, w_curr->phys);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(h11, "I1h11.csv");
    field_visual(h12, "I1h12.csv");
    field_visual(w_curr, "I1w.csv");
    field_visual(u, "I1u.csv");
    field_visual(v, "I1v.csv");
    field_visual(S, "I1S.csv");
    field_visual(w_curr, "I1w.csv");
    field_visual(r1_curr, "I1r1.csv");
    field_visual(r2_curr, "I1r2.csv");
    r1nonl_func(r1nonl, u, v, S, r1_curr, r2_curr, w_curr, h11, lambda, aux, aux1);
    r2nonl_func(r2nonl, u, v, S, r1_curr, r2_curr, w_curr, h12, lambda, aux, aux1); 
    wnonl_func(wnonl, h11,h12,p11,p12,p21,r1_curr,r2_curr,w_curr,u,v,Ra,S,Re, Er, lambda,aux,aux1);
    BwdTrans(mesh,wnonl->spec,wnonl->phys);
    BwdTrans(mesh,r1nonl->spec,r1nonl->phys);
    BwdTrans(mesh,r2nonl->spec,r2nonl->phys);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(wnonl, "I1wnonl.csv");
    field_visual(r1nonl, "I1r1nonl.csv");
    field_visual(r2nonl, "I1r2nonl.csv");

    integrate_func1(w_old, w_curr, w_new, wnonl, wIF, wIFh, dt);
    integrate_func1(r1_old, r1_curr, r1_new, r1nonl, r1IF, r1IFh, dt);
    integrate_func1(r2_old, r2_curr, r2_new, r2nonl, r2IF, r2IFh, dt);
    
    
    ////////////////// Integrate 2 /////////////////////
    curr_func(r1_curr, r2_curr,w_curr,u,v,S, h11, h12);
    BwdTrans(mesh, w_curr->spec, w_curr->phys);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(h11, "I2h11.csv");
    field_visual(h12, "I2h12.csv");
    field_visual(w_curr, "I2w.csv");
    field_visual(u, "I2u.csv");
    field_visual(v, "I2v.csv");
    field_visual(S, "I2S.csv");
    field_visual(w_curr, "I2w.csv");
    field_visual(r1_curr, "I2r1.csv");
    field_visual(r2_curr, "I2r2.csv");
    r1nonl_func(r1nonl, u, v, S, r1_curr, r2_curr, w_curr, h11, lambda, aux, aux1);
    r2nonl_func(r2nonl, u, v, S, r1_curr, r2_curr, w_curr, h12, lambda, aux, aux1); 
    wnonl_func(wnonl, h11,h12,p11,p12,p21,r1_curr,r2_curr,w_curr,u,v,Ra,S,Re, Er, lambda,aux,aux1);
    BwdTrans(mesh,wnonl->spec,wnonl->phys);
    BwdTrans(mesh,r1nonl->spec,r1nonl->phys);
    BwdTrans(mesh,r2nonl->spec,r2nonl->phys);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(wnonl, "I2wnonl.csv");
    field_visual(r1nonl, "I2r1nonl.csv");
    field_visual(r2nonl, "I2r2nonl.csv");
    integrate_func2(w_old, w_curr, w_new, wnonl, wIF, wIFh, dt);
    integrate_func2(r1_old, r1_curr, r1_new, r1nonl, r1IF, r1IFh, dt);
    integrate_func2(r2_old, r2_curr, r2_new, r2nonl, r2IF, r2IFh, dt);
    


    ////////////////// Integrate 3 /////////////////////
    curr_func(r1_curr, r2_curr,w_curr,u,v,S, h11, h12);
    BwdTrans(mesh, w_curr->spec, w_curr->phys);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(h11, "I3h11.csv");
    field_visual(h12, "I3h12.csv");
    field_visual(w_curr, "I3w.csv");
    field_visual(u, "I3u.csv");
    field_visual(v, "I3v.csv");
    field_visual(S, "I3S.csv");
    field_visual(w_curr, "I3w.csv");
    field_visual(r1_curr, "I3r1.csv");
    field_visual(r2_curr, "I3r2.csv");
    r1nonl_func(r1nonl, u, v, S, r1_curr, r2_curr, w_curr, h11, lambda, aux, aux1);
    r2nonl_func(r2nonl, u, v, S, r1_curr, r2_curr, w_curr, h12, lambda, aux, aux1); 
    wnonl_func(wnonl, h11,h12,p11,p12,p21,r1_curr,r2_curr,w_curr,u,v,Ra,S,Re, Er, lambda,aux,aux1);
    BwdTrans(mesh,wnonl->spec,wnonl->phys);
    BwdTrans(mesh,r1nonl->spec,r1nonl->phys);
    BwdTrans(mesh,r2nonl->spec,r2nonl->phys);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(wnonl, "I3wnonl.csv");
    field_visual(r1nonl, "I3r1nonl.csv");
    field_visual(r2nonl, "I3r2nonl.csv");
    integrate_func3(w_old, w_curr, w_new, wnonl, wIF, wIFh, dt);
    integrate_func3(r1_old, r1_curr, r1_new, r1nonl, r1IF, r1IFh, dt);
    integrate_func3(r2_old, r2_curr, r2_new, r2nonl, r2IF, r2IFh, dt);
    


    ////////////////// Integrate 4 /////////////////////
    curr_func(r1_curr, r2_curr,w_curr,u,v,S, h11, h12);
    BwdTrans(mesh, w_curr->spec, w_curr->phys);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(h11, "I4h11.csv");
    field_visual(h12, "I4h12.csv");
    field_visual(w_curr, "I4w.csv");
    field_visual(u, "I4u.csv");
    field_visual(v, "I4v.csv");
    field_visual(S, "I4S.csv");
    field_visual(w_curr, "I4w.csv");
    field_visual(r1_curr, "I4r1.csv");
    field_visual(r2_curr, "I4r2.csv");
    r1nonl_func(r1nonl, u, v, S, r1_curr, r2_curr, w_curr, h11, lambda, aux, aux1);
    r2nonl_func(r2nonl, u, v, S, r1_curr, r2_curr, w_curr, h12, lambda, aux, aux1); 
    wnonl_func(wnonl, h11,h12,p11,p12,p21,r1_curr,r2_curr,w_curr,u,v,Ra,S,Re, Er, lambda,aux,aux1);
    BwdTrans(mesh,wnonl->spec,wnonl->phys);
    BwdTrans(mesh,r1nonl->spec,r1nonl->phys);
    BwdTrans(mesh,r2nonl->spec,r2nonl->phys);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(wnonl, "I4wnonl.csv");
    field_visual(r1nonl, "I4r1nonl.csv");
    field_visual(r2nonl, "I4r2nonl.csv");
    integrate_func4(w_old, w_curr, w_new, wnonl, wIF, wIFh, dt);
    integrate_func4(r1_old, r1_curr, r1_new, r1nonl, r1IF, r1IFh, dt);
    integrate_func4(r2_old, r2_curr, r2_new, r2nonl, r2IF, r2IFh, dt);
    
    SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(w_old->spec, w_new->spec, 
    mesh->Nxh, mesh->Ny, mesh->BSZ);
    SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1_old->spec, r1_new->spec, 
    mesh->Nxh, mesh->Ny, mesh->BSZ);
    SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1_old->spec, r1_new->spec, 
    mesh->Nxh, mesh->Ny, mesh->BSZ);

    return 0;
}