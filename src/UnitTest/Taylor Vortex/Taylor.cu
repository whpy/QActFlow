#include <Basic/QActFlow.h>
#include <Basic/FldOp.cuh>
#include <Basic/Field.h>
#include <Basic/cuComplexBinOp.hpp>
#include <Stream/Streamfunc.cuh>
#include <TimeIntegration/RK4.cuh>
#include <stdlib.h>
#include <iostream>

using namespace std;

// the classical taylor green vortex

__global__ 
void init_func(Qreal* fp, Qreal dx, Qreal dy, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        Qreal x = i*dx;
        Qreal y = j*dy;
        // fp[index] = exp(-1.0* ((x-dx*Nx/2)*(x-dx*Nx/2) + (y-dy*Ny/2)*(y-dy*Ny/2)) );
        // fp[index] = -2*cos(x)*cos(y);
        fp[index] = -40*exp( -1*(pow(x-2*M_PI,2) + pow(y-2*M_PI,2))/0.1 ) 
        + 400*(pow(x-2*M_PI,2) + pow(y-2*M_PI,2))*exp( -1*(pow(x-2*M_PI,2) + pow(y-2*M_PI,2))/0.1 ) ;
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
void wlin_func(Qreal* IFuh, Qreal* IFu, Qreal* k_squared, Qreal Re,
Qreal dt, int Nxh, int Ny, int BSZ)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    Qreal alpha = -1.0/Re *(k_squared[index]) - 0.1;
    if(i<Nxh && j<Ny){
        IFuh[index] = exp( alpha *dt/2);
        IFu[index] = exp( alpha *dt);
    }
}

void wnonl_func(Field* wnonl, Field* wcurr, Field* u, Field *v, Qreal t, Field *aux){
    Mesh* mesh = wnonl->mesh;
    dim3 dimGrid = mesh->dimGridp;
    dim3 dimBlock = mesh->dimBlockp;
    // aux.phys = w*u
    FldMul<<<dimGrid, dimBlock>>>(u->phys, wcurr->phys, 1.0, aux->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    // aux.spec = Dx(w*u)
    xDeriv(aux->spec, aux->spec, mesh);

    // wnonl.phys = w*v
    FldMul<<<dimGrid, dimBlock>>>(v->phys, wcurr->phys, 1.0, wnonl->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    // wnonl.spec = Dy(w*v)
    yDeriv(wnonl->spec, wnonl->spec, mesh);

    SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(1.0, aux->spec, 1.0, wnonl->spec, wnonl->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
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
    int Ns = 2001;
    int Nx = 512; // same as colin
    int Ny = 512;
    int Nxh = Nx/2+1;
    Qreal Lx = 4*M_PI;
    Qreal Ly = 4*M_PI;
    Qreal dx = Lx/Nx;
    Qreal dy = Ly/Ny;
    Qreal dt = 0.02; // same as colin
    // Qreal a = 1.0;

    Qreal Re = 0.1;

    // Fldset test
    // vortex fields
    Mesh *mesh = new Mesh(BSZ, Nx, Ny, Lx, Ly);
    Field *wold = new Field(mesh); Field *wcurr = new Field(mesh); Field *wnew = new Field(mesh);
    Field *wnonl = new Field(mesh);

    // velocity fields
    Field *u = new Field(mesh);
    Field *v = new Field(mesh);
    
    // time integrating factors
    Qreal *IFw, *IFwh;
    cudaMallocManaged(&IFw, sizeof(Qreal)*Nxh*Ny);
    cudaMallocManaged(&IFwh, sizeof(Qreal)*Nxh*Ny);

    // auxiliary field
    Field *aux = new Field(mesh);

    coord(*mesh);
    // int m = 0;
    // initialize the field
    // set up the Integrating factor
    // we may take place here by IF class
    wlin_func<<<mesh->dimGridsp,mesh->dimBlocksp>>>(IFwh, IFw, mesh->k_squared, Re, 
    dt, mesh->Nxh, mesh->Ny, mesh->BSZ);
    // initialize the physical space of w
    init_func<<<mesh->dimGridp,mesh->dimBlockp>>>(wold->phys, 
    mesh->dx, mesh->dy, mesh->Nx, mesh->Ny, mesh->BSZ);
    // initialize the spectral space of w 
    FwdTrans(mesh, wold->phys, wold->spec);


    
    for(int m=0 ;m<Ns ;m++){
        integrate_func0(wold, wcurr, wnew, IFw, IFwh, dt);
        
        BwdTrans(mesh, wcurr->spec, wcurr->phys);
        vel_func(wcurr, u, v);
        wnonl_func(wnonl, wcurr, u, v, m*dt, aux);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func1(wold, wcurr, wnew, wnonl, IFw, IFwh, dt);
        
        BwdTrans(mesh, wcurr->spec, wcurr->phys);
        vel_func(wcurr, u, v);
        wnonl_func(wnonl, wcurr, u, v, m*dt, aux);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func2(wold, wcurr, wnew, wnonl, IFw, IFwh, dt);
        
        BwdTrans(mesh, wcurr->spec, wcurr->phys);
        vel_func(wcurr, u, v);
        wnonl_func(wnonl, wcurr, u, v, m*dt, aux);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func3(wold, wcurr, wnew, wnonl, IFw, IFwh, dt);
        
        BwdTrans(mesh, wcurr->spec, wcurr->phys);
        vel_func(wcurr, u, v);
        wnonl_func(wnonl, wcurr, u, v, m*dt, aux);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func4(wold, wcurr, wnew, wnonl, IFw, IFwh, dt);
        
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wnew->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
        
        if(m%1 == 0) cout << "t = " << m*dt << endl;
        if (m%20 == 0){
            BwdTrans(mesh, wold->spec, wold->phys);
            cuda_error_func( cudaDeviceSynchronize() );
            field_visual(wold, to_string(m)+"w.csv");
            // printf("t: %f    val:%.8f   exa:%.8f    err: %.8f \n",m*dt,  u->phys[5],exact((m)*dt), u->phys[5]-exact((m)*dt));
            // cout<<"t: " << m*dt << "  " << u->phys[5] << endl;
        }
    }

    delete mesh;
    
    return 0;
}
