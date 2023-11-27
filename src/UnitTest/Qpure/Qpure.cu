#include <Basic/QActFlow.h>
#include <Basic/FldOp.cuh>
#include <Basic/Field.h>
#include <Basic/cuComplexBinOp.hpp>
#include <Stream/Streamfunc.cuh>
#include <TimeIntegration/RK4.cuh>

#include <stdlib.h>
#include <iostream>
#include <random>

using namespace std;

// the classical taylor green vortex

void init_func(Qreal* r1, Qreal* r2, Qreal dx, Qreal dy, int Nx, int Ny, int BSZ){
    for (int i=0; i<Nx; i++){
        for(int j=0; j<Ny; j++){
            int index = i + j*Nx;
            Qreal x = i*dx;
            Qreal y = j*dy;
            // fp[index] = exp(-1.0* ((x-dx*Nx/2)*(x-dx*Nx/2) + (y-dy*Ny/2)*(y-dy*Ny/2)) );
            // fp[index] = -2*cos(x)*cos(y);
            // r1[index] = 0.2*(sin(x)*sin(x) - 0.5);
            // r2[index] = 0.2*sin(x)*cos(x);
            r1[index] = 0.2*(sin(x+y)*sin(x+y) - 0.5);
            r2[index] = 0.2*sin((x+y))*cos((x+y));
        }
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

__global__
void Mr1lin_func(Qreal* IFr1h, Qreal* IFr1, Qreal* k_squared, Qreal Re,
Qreal dt, int Nxh, int Ny, int BSZ)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    Qreal alpha = -1.0*(k_squared[index]) + 1.0;
    if(i<Nxh && j<Ny){
        IFr1h[index] = exp( alpha *dt/2);
        IFr1[index] = exp( alpha *dt);
    }
}

__global__
void Mr2lin_func(Qreal* IFr1h, Qreal* IFr1, Qreal* k_squared, Qreal Re,
Qreal dt, int Nxh, int Ny, int BSZ)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    Qreal alpha = -1.0*(k_squared[index]) + 1.0;
    if(i<Nxh && j<Ny){
        IFr1h[index] = exp( alpha *dt/2);
        IFr1[index] = exp( alpha *dt);
    }
}

void r1nonl_func(Field* r1nonl, Field* r1curr, Field* r2curr, Field* S, Qreal t, Field *aux){
    Mesh* mesh = r1nonl->mesh;
    dim3 dimGrid = mesh->dimGridp;
    dim3 dimBlock = mesh->dimBlockp;
    BwdTrans(mesh, r1curr->spec, r1curr->phys);
    BwdTrans(mesh, r2curr->spec, r2curr->phys);
    // nonl = -S^2 r1
    //cal the S
    S_func(r1curr, r2curr, S);

    FldMul<<<dimGrid, dimBlock>>>(S->phys,S->phys,-1.0,r1nonl->phys,mesh->Nx, mesh->Ny, mesh->BSZ);
    FldMul<<<dimGrid, dimBlock>>>(r1nonl->phys,r1curr->phys,1.0,r1nonl->phys,mesh->Nx, mesh->Ny, mesh->BSZ);

    FwdTrans(mesh, r1nonl->phys, r1nonl->spec);
    
}

void r2nonl_func(Field* r2nonl, Field* r1curr, Field* r2curr, Field* S, Qreal t, Field *aux){
    Mesh* mesh = r2nonl->mesh;
    dim3 dimGrid = mesh->dimGridp;
    dim3 dimBlock = mesh->dimBlockp;
    BwdTrans(mesh, r1curr->spec, r1curr->phys);
    BwdTrans(mesh, r2curr->spec, r2curr->phys);
    // nonl = -S^2 r2
    //cal the S
    S_func(r1curr, r2curr, S);

    FldMul<<<dimGrid, dimBlock>>>(S->phys, S->phys, -1.0, r2nonl->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    FldMul<<<dimGrid, dimBlock>>>(r2nonl->phys, r1curr->phys, 1.0, r2nonl->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

    FwdTrans(mesh, r2nonl->phys, r2nonl->spec);
    
}

void Or1nonl_func(Field *r1nonl, Field *aux, Field *r1, Field *r2, Field *w, 
                        Field *u, Field *v, Field *S, Qreal lambda, Qreal cn, Qreal Pe){
    // Mesh* mesh = r1nonl->mesh;
    // dim3 dimGrid = mesh->dimGridp;
    // dim3 dimBlock = mesh->dimBlockp;
    // BwdTrans(mesh, r1->spec, r1->phys);
    // cuda_error_func( cudaDeviceSynchronize() );
    // BwdTrans(mesh, r2->spec, r2->phys);
    // cuda_error_func( cudaDeviceSynchronize() );
    // // nonl = -S^2 r1
    // //cal the S
    // S_func(r1, r2, S);
    // cuda_error_func( cudaDeviceSynchronize() );

    // FldMul<<<dimGrid, dimBlock>>>(S->phys,S->phys,-1.0,r1nonl->phys,mesh->Nx, mesh->Ny, mesh->BSZ);
    // cuda_error_func( cudaDeviceSynchronize() );
    // FldMul<<<dimGrid, dimBlock>>>(r1nonl->phys,r1->phys,1.0,r1nonl->phys,mesh->Nx, mesh->Ny, mesh->BSZ);
    // cuda_error_func( cudaDeviceSynchronize() );

    // FwdTrans(mesh, r1nonl->phys, r1nonl->spec);
    // cuda_error_func( cudaDeviceSynchronize() );
    
    
    BwdTrans(r1->mesh,r1->spec, r1->phys);
    BwdTrans(r2->mesh,r2->spec, r2->phys);
    BwdTrans(w->mesh, w->spec, w->phys);
    vel_func(w, u, v);
    S_func(r1, r2, S);
    Mesh *mesh = r1nonl->mesh;
    int Nx = mesh->Nx; int Ny = mesh->Ny; int BSZ = mesh->BSZ;
    dim3 dimGrid = mesh->dimGridp;
    dim3 dimBlock = mesh->dimBlockp;

    FldMul<<<dimGrid, dimBlock>>>(S->phys, S->phys, -1.0, r1nonl->phys,Nx, Ny, BSZ);
    FldMul<<<dimGrid, dimBlock>>>(r1->phys, r1nonl->phys, 1.0, r1nonl->phys,Nx, Ny, BSZ);
    FwdTrans(r1nonl->mesh, r1nonl->phys, r1nonl->spec);
}

void Mr1nonl_func(Field* r1nonl, Field* r1curr, Field* r2curr, Field* S, Qreal t, Field *aux){
    Mesh* mesh = r1nonl->mesh;
    dim3 dimGrid = mesh->dimGridp;
    dim3 dimBlock = mesh->dimBlockp;
    BwdTrans(mesh, r1curr->spec, r1curr->phys);
    BwdTrans(mesh, r2curr->spec, r2curr->phys);
    // nonl = -S^2 r1
    //cal the S
    S_func(r1curr, r2curr, S);

    FldMul<<<dimGrid, dimBlock>>>(S->phys,S->phys,-1.0,r1nonl->phys,mesh->Nx, mesh->Ny, mesh->BSZ);
    FldMul<<<dimGrid, dimBlock>>>(r1nonl->phys,r1curr->phys,1.0,r1nonl->phys,mesh->Nx, mesh->Ny, mesh->BSZ);

    FwdTrans(mesh, r1nonl->phys, r1nonl->spec);
    
}

void Or2nonl_func(Field *r2nonl, Field *aux, Field *r1, Field *r2, Field *w, 
                        Field *u, Field *v, Field *S, Qreal lambda, Qreal cn, Qreal Pe){
    Mesh* mesh = r2nonl->mesh;
    dim3 dimGridp = mesh->dimGridp;
    dim3 dimBlockp = mesh->dimBlockp;
    // BwdTrans(mesh, r1->spec, r1->phys);
    // cuda_error_func( cudaDeviceSynchronize() );
    // BwdTrans(mesh, r2->spec, r2->phys);
    // cuda_error_func( cudaDeviceSynchronize() );
    // // nonl = -S^2 r2
    // //cal the S
    // S_func(r1, r2, S);
    // cuda_error_func( cudaDeviceSynchronize() );

    // FldMul<<<dimGridp, dimBlockp>>>(S->phys, S->phys, -1.0, r2nonl->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    // cuda_error_func( cudaDeviceSynchronize() );
    // FldMul<<<dimGridp, dimBlockp>>>(r2nonl->phys, r2->phys, 1.0, r2nonl->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    // cuda_error_func( cudaDeviceSynchronize() );

    // FwdTrans(mesh, r2nonl->phys, r2nonl->spec);
    // cuda_error_func( cudaDeviceSynchronize() );

    BwdTrans(r1->mesh,r1->spec, r1->phys);
    BwdTrans(r2->mesh,r2->spec, r2->phys);
    BwdTrans(w->mesh, w->spec, w->phys);
    vel_func(w, u, v);
    S_func(r1,r2,S);

    FldMul<<<dimGridp, dimBlockp>>>(S->phys, S->phys, -1.0, r2nonl->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    FldMul<<<dimGridp, dimBlockp>>>(r2->phys, r2nonl->phys, 1.0, r2nonl->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    FwdTrans(r2nonl->mesh, r2nonl->phys, r2nonl->spec);
}

void Mr2nonl_func(Field* r2nonl, Field* r1curr, Field* r2curr, Field* S, Qreal t, Field *aux){
    Mesh* mesh = r2nonl->mesh;
    dim3 dimGrid = mesh->dimGridp;
    dim3 dimBlock = mesh->dimBlockp;
    BwdTrans(mesh, r1curr->spec, r1curr->phys);
    BwdTrans(mesh, r2curr->spec, r2curr->phys);
    // nonl = -S^2 r2
    //cal the S
    S_func(r1curr, r2curr, S);

    FldMul<<<dimGrid, dimBlock>>>(S->phys, S->phys, -1.0, r2nonl->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    FldMul<<<dimGrid, dimBlock>>>(r2nonl->phys, r2curr->phys, 1.0, r2nonl->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

    FwdTrans(mesh, r2nonl->phys, r2nonl->spec);
    
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
            double x = mesh.dx*i;
            double y = mesh.dy*j;
            xcoord << x << ",";
            ycoord << y << ",";
        }
        xcoord << endl;
        ycoord << endl;
    }
    xcoord.close();
    ycoord.close();
}
// we test the performance of the RK4 on linear ODE that du/dt = -u where
// the exact solution should be u = c0*exp(-t), c0 depends on initial conditon.

int main(){
    int BSZ = 16;
    int Ns = 8001;
    int Nx = 512; // same as colin
    int Ny = 512;
    int Nxh = Nx/2+1;
    Qreal Lx = 33* 2*M_PI;
    Qreal Ly = 33* 2*M_PI;
    Qreal dx = Lx/Nx;
    Qreal dy = Ly/Ny;
    Qreal dt = 0.01; // same as colin
    // Qreal a = 1.0;

    Qreal Re = 0.1;
    Qreal lambda = 0.1;
    Qreal cn = 1.0;
    Qreal Pe = 1.0;

    // Fldset test
    // vortex fields
    Mesh *mesh = new Mesh(BSZ, Nx, Ny, Lx, Ly);
    cout << "Re = " << Re << "  " << endl;
    cout << "Pe = " << Pe << "  "<< endl;
   cout << "cn = " << cn << "  "; cout << "dt = " << dt << endl;
    cout << "Lx = " << mesh->Lx << " "<< "Ly = " << mesh->Ly << " " << endl;
    cout << "Nx = " << mesh->Nx << " "<< "Ny = " << mesh->Ny << " " << endl;
    cout << "dx = " << mesh->dx << " "<< "dy = " << mesh->dy << " " << endl;
    cout << "Nx*dx = " << mesh->Nx*mesh->dx << " "<< "Ny*dy = " << mesh->Ny*mesh->dy << " " << endl;
    cout << "End time: Ns*dt = " << Ns*dt << endl;
    Field *r1old = new Field(mesh); 
    Field *r1curr = new Field(mesh); 
    Field *r1new = new Field(mesh);
    Field *r1nonl = new Field(mesh);

    Field *r2old = new Field(mesh); 
    Field *r2curr = new Field(mesh); 
    Field *r2new = new Field(mesh);
    Field *r2nonl = new Field(mesh);

    Field *S = new Field(mesh);
    
    Field* wcurr = new Field(mesh);
    Field* wold = new Field(mesh);
    Field* wnew = new Field(mesh);
    Field* u = new Field(mesh);
    Field *v = new Field(mesh);
    // time integrating factors
    Qreal *IFr1, *IFr1h;
    cudaMallocManaged(&IFr1, sizeof(Qreal)*Nxh*Ny);
    cudaMallocManaged(&IFr1h, sizeof(Qreal)*Nxh*Ny);

    Qreal *IFr2, *IFr2h;
    cudaMallocManaged(&IFr2, sizeof(Qreal)*Nxh*Ny);
    cudaMallocManaged(&IFr2h, sizeof(Qreal)*Nxh*Ny);

    // intermediate fields
    
    Field *p11 = new Field(mesh); Field *p12 = new Field(mesh); Field* p21 = new Field(mesh);
    
    // auxiliary fields
    Field *aux = new Field(mesh); Field *aux1 = new Field(mesh); 

    coord(*mesh);
    // int m = 0;
    // initialize the field
    // set up the Integrating factor
    // we may take place here by IF class
    Mr1lin_func<<<mesh->dimGridsp,mesh->dimBlocksp>>>(IFr1h, IFr1, mesh->k_squared, Re, 
    dt, mesh->Nxh, mesh->Ny, mesh->BSZ);
    Mr2lin_func<<<mesh->dimGridsp,mesh->dimBlocksp>>>(IFr2h, IFr2, mesh->k_squared, Re, 
    dt, mesh->Nxh, mesh->Ny, mesh->BSZ);
    // initialize the physical space of Q
    init_func(r1old->phys, r2old->phys, mesh->dx, mesh->dy, mesh->Nx, mesh->Ny, mesh->BSZ);
    // initialize the spectral space of w 
    FwdTrans(mesh, r1old->phys, r1old->spec);
    FwdTrans(mesh, r2old->phys, r2old->spec);

    BwdTrans(mesh, r1old->spec, r1old->phys);
    BwdTrans(mesh, r2old->spec, r2old->phys);
    S_func(r1old, r2old, S);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(r1old, "00r1.csv");
    field_visual(r2old, "00r2.csv");
    field_visual(S, "00S.csv");
    field_visual(r1nonl, "00r1nonl.csv");
    field_visual(r2nonl, "00r2nonl.csv");
    for(int m=0 ;m<Ns ;m++){
        integrate_func0(r1old, r1curr, r1new, IFr1, IFr1h, dt);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func0(r2old, r2curr, r1new, IFr2, IFr2h, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        // Mr1nonl_func(r1nonl, r1curr, r2curr, S, m*dt, aux);
        // Mr2nonl_func(r2nonl, r1curr, r2curr, S, m*dt, aux);
        Or1nonl_func(r1nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        Or2nonl_func(r2nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func1(r1old, r1curr, r1new, r1nonl, IFr1, IFr1h, dt);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func1(r2old, r2curr, r2new, r2nonl, IFr2, IFr2h, dt);
        cuda_error_func( cudaDeviceSynchronize() );
        
        // Mr1nonl_func(r1nonl, r1curr, r2curr, S, m*dt, aux);
        // Mr2nonl_func(r2nonl, r1curr, r2curr, S, m*dt, aux);
        Or1nonl_func(r1nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        Or2nonl_func(r2nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func2(r1old, r1curr, r1new, r1nonl, IFr1, IFr1h, dt);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func2(r2old, r2curr, r2new, r2nonl, IFr2, IFr2h, dt);
        
        // Mr1nonl_func(r1nonl, r1curr, r2curr, S, m*dt, aux);
        // Mr2nonl_func(r2nonl, r1curr, r2curr, S, m*dt, aux);
        Or1nonl_func(r1nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        Or2nonl_func(r2nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func3(r1old, r1curr, r1new, r1nonl, IFr1, IFr1h, dt);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func3(r2old, r2curr, r2new, r2nonl, IFr2, IFr2h, dt);
        
        // Mr1nonl_func(r1nonl, r1curr, r2curr, S, m*dt, aux);
        // Mr2nonl_func(r2nonl, r1curr, r2curr, S, m*dt, aux);
        Or1nonl_func(r1nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        Or2nonl_func(r2nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func4(r1old, r1curr, r1new, r1nonl, IFr1, IFr1h, dt);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func4(r2old, r2curr, r2new, r2nonl, IFr2, IFr2h, dt);
        cuda_error_func( cudaDeviceSynchronize() );
        
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1new->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
        cuda_error_func( cudaDeviceSynchronize() );
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2new->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
        cuda_error_func( cudaDeviceSynchronize() );

        if(m%10 == 0) {cout << "\r" << "t = " << m*dt << flush;}
        if (m%80 == 0){
            BwdTrans(mesh, r1old->spec, r1old->phys);
            cuda_error_func( cudaDeviceSynchronize() );
            BwdTrans(mesh, r2old->spec, r2old->phys);
            cuda_error_func( cudaDeviceSynchronize() );
            S_func(r1old, r2old, S);
            cuda_error_func( cudaDeviceSynchronize() );
            field_visual(r1old, to_string(m)+"r1.csv");
            field_visual(r2old, to_string(m)+"r2.csv");
            field_visual(r1nonl, to_string(m)+"r1nonl.csv");
            field_visual(r2nonl, to_string(m)+"r2nonl.csv");
            field_visual(S, to_string(m)+"S.csv");
            // printf("t: %f    val:%.8f   exa:%.8f    err: %.8f \n",m*dt,  u->phys[5],exact((m)*dt), u->phys[5]-exact((m)*dt));
            // cout<<"t: " << m*dt << "  " << u->phys[5] << endl;
        }
    }

    delete mesh;
    
    return 0;
}