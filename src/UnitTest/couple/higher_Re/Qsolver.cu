#include "Qsolver.cuh"
using std::string;
using std::cout; 
using std::endl;

int main(){
    int startpoint = 0;
    eInitType Init = Phy_init;
    cout << "start point = " << startpoint << endl;
    cout << "initialization way: " << InitType[Init] << endl;
    int BSZ = 16;
    int Ns = 150001;
    int Nx = 512; // same as colin
    int Ny = 512;
    int Nxh = Nx/2+1;
    Qreal Lx = 33* 2*M_PI;
    Qreal Ly = Lx;
    Qreal dx = Lx/Nx;
    Qreal dy = Ly/Ny;
    Qreal dt = 0.0001; // same as colin
    // Qreal a = 1.0;

    Qreal Re = 1.0;
    Qreal Pe = 1.0;
    Qreal Er = 0.1;
    Qreal Ra = -0.2;
    Qreal cf = sqrt(0.000075);
    Qreal cn = 1.0;
    Qreal lambda = 0.1;
    
    // main Fields to be solved
    // *_curr act as an intermediate while RK4 timeintegration
    // *_new store the value of next time step 

    Mesh *mesh = new Mesh(BSZ, Nx, Ny, Lx, Ly);
    coord(*mesh);
    
    cout << "Re = " << Re << "  "; cout << "Er = " << Er << "   "; 
    cout << "Pe = " << Pe << "  "; cout << "Ra = " << Ra << endl;
    cout << "cf = " << cf << "  "; cout << "cn = " << cn << "  "; cout << "dt = " << dt << endl;
    cout << "Lx = " << mesh->Lx << " "<< "Ly = " << mesh->Ly << " " << "   ";
    cout << "Nx = " << mesh->Nx << " "<< "Ny = " << mesh->Ny << " " << endl;
    cout << "dx = " << mesh->dx << " "<< "dy = " << mesh->dy << " " << " ";
    cout << "dt = " << dt << endl;
    cout << "Nx*dx = " << mesh->Nx*mesh->dx << " "<< "Ny*dy = " << mesh->Ny*mesh->dy << " " << endl;
    cout << "End time: Ns*dt = " << Ns*dt << endl;
    
    // Fields to be solved
    Field *wold = new Field(mesh); 
    Field *wcurr = new Field(mesh); 
    Field *wnew = new Field(mesh);
    Field *wnonl = new Field(mesh); 
    
    Field *r1old = new Field(mesh); 
    Field *r1curr = new Field(mesh); 
    Field *r1new = new Field(mesh);
    Field *r1nonl = new Field(mesh); 
    
    Field *r2old = new Field(mesh); 
    Field *r2curr = new Field(mesh); 
    Field *r2new = new Field(mesh);
    Field *r2nonl = new Field(mesh);
    
    // linear factors
    Qreal *IFw, *IFwh; 
    cudaMallocManaged(&IFw, sizeof(Qreal)*Nxh*Ny); 
    cudaMallocManaged(&IFwh, sizeof(Qreal)*Nxh*Ny);

    Qreal *IFr1, *IFr1h;
    cudaMallocManaged(&IFr1, sizeof(Qreal)*Nxh*Ny);
    cudaMallocManaged(&IFr1h, sizeof(Qreal)*Nxh*Ny);

    Qreal *IFr2, *IFr2h;
    cudaMallocManaged(&IFr2, sizeof(Qreal)*Nxh*Ny);
    cudaMallocManaged(&IFr2h, sizeof(Qreal)*Nxh*Ny);


    // intermediate fields
    Field *u = new Field(mesh); 
    Field *v = new Field(mesh); 
    Field *S = new Field(mesh);
    Field *p11 = new Field(mesh); 
    Field *p12 = new Field(mesh); 
    Field* p21 = new Field(mesh);
    
    // auxiliary fields
    Field *aux = new Field(mesh); Field *aux1 = new Field(mesh); 

    // field \alpha to be modified (scalar at the very first)
    Field *alpha = new Field(mesh);

    // decouple variables
    Field * r1zero = new Field(mesh);
    FldSet<<<mesh->dimGridp,mesh->dimBlockp>>>(r1zero->phys, 0.0, mesh->Nx, mesh->Ny, mesh->BSZ);
    SpecSet<<<mesh->dimGridsp,mesh->dimBlocksp>>>(r1zero->spec, make_cuDoubleComplex(0.0,0.0), mesh->Nxh, mesh->Ny, mesh->BSZ);
    
    Field * r2zero = new Field(mesh);
    FldSet<<<mesh->dimGridp,mesh->dimBlockp>>>(r2zero->phys, 0.0, mesh->Nx, mesh->Ny, mesh->BSZ);
    SpecSet<<<mesh->dimGridsp,mesh->dimBlocksp>>>(r2zero->spec, make_cuDoubleComplex(0.0,0.0), mesh->Nxh, mesh->Ny, mesh->BSZ);
    
    Field * wzero = new Field(mesh);
    FldSet<<<mesh->dimGridp,mesh->dimBlockp>>>(wzero->phys, 0.0, mesh->Nx, mesh->Ny, mesh->BSZ);
    SpecSet<<<mesh->dimGridsp,mesh->dimBlocksp>>>(wzero->spec, make_cuDoubleComplex(0.0,0.0), mesh->Nxh, mesh->Ny, mesh->BSZ);

    //////////////////////// precomputation //////////////////////////
    r1lin_func<<<mesh->dimGridsp,mesh->dimBlocksp>>>(IFr1h, IFr1, mesh->k_squared, 
    Pe, cn, dt, mesh->Nxh, mesh->Ny, mesh->BSZ);
    r2lin_func<<<mesh->dimGridsp,mesh->dimBlocksp>>>(IFr2h, IFr2, mesh->k_squared, 
    Pe, cn, dt, mesh->Nxh, mesh->Ny, mesh->BSZ);
    cuda_error_func( cudaDeviceSynchronize() );

    wlin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(IFwh, IFw, mesh->k_squared, Re, cf, dt, mesh->Nxh, mesh->Ny, mesh->BSZ);
    cuda_error_func( cudaDeviceSynchronize() );
    
    // the precomputation function also updates the spectrum of corresponding variables
    precompute_func(r1old, r2old, wold, Init);
    alpha_init(alpha->phys, Ra, dx, dy, Nx, Ny);
    FwdTrans(mesh, alpha->phys, alpha->spec);
    
    // prepare the referenced system, record the initial conditions(verify whether
    // spectral and physical space initialized simultaneously)
    field_visual(alpha, "alphastart.csv");

    BwdTrans(mesh, wold->spec, wold->phys);
    BwdTrans(mesh, r1old->spec, r1old->phys);
    BwdTrans(mesh, r2old->spec, r2old->phys);
    BwdTrans(mesh, alpha->spec, alpha->phys);
    S_func(r1old, r2old, S);
    cuda_error_func( cudaDeviceSynchronize() );
    
    field_visual(wold,"wstart.csv");
    field_visual(r1old,"r1start.csv");
    field_visual(r2old,"r2start.csv");
    field_visual(S,"Sstart.csv");
    field_visual(alpha, "alphastart.csv");
    
    //////////////////////// time stepping //////////////////////////
    for(int m=0 ;m<Ns ;m++){
        // RK4 step 0
        integrate_func0(r1old, r1curr, r1new, IFr1, IFr1h, dt);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func0(r2old, r2curr, r1new, IFr2, IFr2h, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        integrate_func0(wold, wcurr, wnew, IFw, IFwh, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        // RK4 step 1
        vel_func(wcurr, u, v);
        BwdTrans(r1curr->mesh,r1curr->spec, r1curr->phys);
        cuda_error_func( cudaDeviceSynchronize() );
        BwdTrans(r2curr->mesh,r2curr->spec, r2curr->phys);
        cuda_error_func( cudaDeviceSynchronize() );
        BwdTrans(wcurr->mesh, wcurr->spec, wcurr->phys);
        cuda_error_func( cudaDeviceSynchronize() );
        S_func(r1curr,r2curr,S);
        cuda_error_func( cudaDeviceSynchronize() );

        r1nonl_func(r1nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func1(r1old, r1curr, r1new, r1nonl, IFr1, IFr1h, dt);
        cuda_error_func( cudaDeviceSynchronize() );
        
        r2nonl_func(r2nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func1(r2old, r2curr, r2new, r2nonl, IFr2, IFr2h, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        wnonl_func(wnonl, aux, aux1, p11, p12, p21, r1curr, r2curr, wcurr, u, v, alpha, S, Re, Er, cn, lambda);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func1(wold, wcurr, wnew, wnonl, IFw, IFwh, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        // RK4 step 2
        vel_func(wcurr, u, v);
        BwdTrans(r1curr->mesh,r1curr->spec, r1curr->phys);
        cuda_error_func( cudaDeviceSynchronize() );
        BwdTrans(r2curr->mesh,r2curr->spec, r2curr->phys);
        cuda_error_func( cudaDeviceSynchronize() );
        BwdTrans(wcurr->mesh, wcurr->spec, wcurr->phys);
        cuda_error_func( cudaDeviceSynchronize() );
        S_func(r1curr,r2curr,S);
        cuda_error_func( cudaDeviceSynchronize() );

        r1nonl_func(r1nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func2(r1old, r1curr, r1new, r1nonl, IFr1, IFr1h, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        r2nonl_func(r2nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func2(r2old, r2curr, r2new, r2nonl, IFr2, IFr2h, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        wnonl_func(wnonl, aux, aux1, p11, p12, p21, r1curr, r2curr, wcurr, u, v, alpha, S, Re, Er, cn, lambda);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func2(wold, wcurr, wnew, wnonl, IFw, IFwh, dt);
        cuda_error_func( cudaDeviceSynchronize() );
        
        // RK4 step 3
        vel_func(wcurr, u, v);
        BwdTrans(r1curr->mesh,r1curr->spec, r1curr->phys);
        cuda_error_func( cudaDeviceSynchronize() );
        BwdTrans(r2curr->mesh,r2curr->spec, r2curr->phys);
        cuda_error_func( cudaDeviceSynchronize() );
        BwdTrans(wcurr->mesh, wcurr->spec, wcurr->phys);
        cuda_error_func( cudaDeviceSynchronize() );
        S_func(r1curr,r2curr,S);
        cuda_error_func( cudaDeviceSynchronize() );

        r1nonl_func(r1nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func3(r1old, r1curr, r1new, r1nonl, IFr1, IFr1h, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        r2nonl_func(r2nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func3(r2old, r2curr, r2new, r2nonl, IFr2, IFr2h, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        wnonl_func(wnonl, aux, aux1, p11, p12, p21, r1curr, r2curr, wcurr, u, v, alpha, S, Re, Er, cn, lambda);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func3(wold, wcurr, wnew, wnonl, IFw, IFwh, dt);
        cuda_error_func( cudaDeviceSynchronize() );
        
        // RK4 step4
        vel_func(wcurr, u, v);
        BwdTrans(r1curr->mesh,r1curr->spec, r1curr->phys);
        cuda_error_func( cudaDeviceSynchronize() );
        BwdTrans(r2curr->mesh,r2curr->spec, r2curr->phys);
        cuda_error_func( cudaDeviceSynchronize() );
        BwdTrans(wcurr->mesh, wcurr->spec, wcurr->phys);
        cuda_error_func( cudaDeviceSynchronize() );
        S_func(r1curr,r2curr,S);
        cuda_error_func( cudaDeviceSynchronize() );

        r1nonl_func(r1nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func4(r1old, r1curr, r1new, r1nonl, IFr1, IFr1h, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        r2nonl_func(r2nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func4(r2old, r2curr, r2new, r2nonl, IFr2, IFr2h, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        wnonl_func(wnonl, aux, aux1, p11, p12, p21, r1curr, r2curr, wcurr, u, v, alpha, S, Re, Er, cn, lambda);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func4(wold, wcurr, wnew, wnonl, IFw, IFwh, dt);
        
        // update the current fields (spectrally)
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1new->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
        cuda_error_func( cudaDeviceSynchronize() );
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2new->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
        cuda_error_func( cudaDeviceSynchronize() );
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wnew->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);

        if(m%10 == 0) {cout << "\r" << "t = " << (startpoint+m)*dt << flush;}
        if (m%40 == 0){
            BwdTrans(mesh, r1old->spec, r1old->phys);
            cuda_error_func( cudaDeviceSynchronize() );
            BwdTrans(mesh, r2old->spec, r2old->phys);
            cuda_error_func( cudaDeviceSynchronize() );
            S_func(r1old, r2old, S);
            BwdTrans(mesh, wold->spec, wold->phys);
            cuda_error_func( cudaDeviceSynchronize() );
            field_visual(r1old, to_string(m+startpoint)+"r1.csv");
            field_visual(r2old, to_string(m+startpoint)+"r2.csv");
            field_visual(r1nonl, to_string(m+startpoint)+"r1nonl.csv");
            field_visual(r2nonl, to_string(m+startpoint)+"r2nonl.csv");
            field_visual(S, to_string(m+startpoint)+"S.csv");
            field_visual(wold, to_string(m+startpoint)+"w.csv");
        }
    }

    delete mesh;

    return 0;
}

