#include "Qsolver.cuh"
using std::string;
using std::cout; 
using std::endl;


// __global__
// void Mr1lin_func(Qreal* IFr1h, Qreal* IFr1, Qreal* k_squared,
// Qreal dt, int Nxh, int Ny, int BSZ)
// {
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = j*Nxh + i;
//     Qreal alpha = -1.0*(k_squared[index]) + 1.0;
//     if(i<Nxh && j<Ny){
//         IFr1h[index] = exp( alpha *dt/2);
//         IFr1[index] = exp( alpha *dt);
//     }
// }

// __global__
// void Mr2lin_func(Qreal* IFr2h, Qreal* IFr2, Qreal* k_squared,
// Qreal dt, int Nxh, int Ny, int BSZ)
// {
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = j*Nxh + i;
//     Qreal alpha = -1.0*(k_squared[index]) + 1.0;
//     if(i<Nxh && j<Ny){
//         IFr2h[index] = exp( alpha *dt/2);
//         IFr2[index] = exp( alpha *dt);
//     }
// }

// void Or1nonl_func(Field *r1nonl, Field *aux, Field *r1, Field *r2, Field *w, 
//                         Field *u, Field *v, Field *S, Qreal lambda, Qreal cn, Qreal Pe){
//     BwdTrans(r1->mesh,r1->spec, r1->phys);
//     BwdTrans(r2->mesh,r2->spec, r2->phys);
//     BwdTrans(w->mesh, w->spec, w->phys);
//     vel_func(w, u, v);
//     S_func(r1, r2, S);
//     Mesh *mesh = r1nonl->mesh;
//     int Nx = mesh->Nx; int Ny = mesh->Ny; int BSZ = mesh->BSZ;
//     dim3 dimGrid = mesh->dimGridp;
//     dim3 dimBlock = mesh->dimBlockp;

//     FldMul<<<dimGrid, dimBlock>>>(S->phys, S->phys, -1.0, r1nonl->phys,Nx, Ny, BSZ);
//     FldMul<<<dimGrid, dimBlock>>>(r1->phys, r1nonl->phys, 1.0, r1nonl->phys,Nx, Ny, BSZ);
//     FwdTrans(r1nonl->mesh, r1nonl->phys, r1nonl->spec);
// }

// void Mr1nonl_func(Field* r1nonl, Field* r1curr, Field* r2curr, Field* S, Qreal t, Field *aux){
//     Mesh* mesh = r1nonl->mesh;
//     dim3 dimGrid = mesh->dimGridp;
//     dim3 dimBlock = mesh->dimBlockp;
//     BwdTrans(mesh, r1curr->spec, r1curr->phys);
//     BwdTrans(mesh, r2curr->spec, r2curr->phys);
//     // nonl = -S^2 r1
//     //cal the S
//     S_func(r1curr, r2curr, S);

//     FldMul<<<dimGrid, dimBlock>>>(S->phys,S->phys,-1.0,r1nonl->phys,mesh->Nx, mesh->Ny, mesh->BSZ);
//     FldMul<<<dimGrid, dimBlock>>>(r1nonl->phys,r1curr->phys,1.0,r1nonl->phys,mesh->Nx, mesh->Ny, mesh->BSZ);

//     FwdTrans(mesh, r1nonl->phys, r1nonl->spec);
    
// }

// void Or2nonl_func(Field *r2nonl, Field *aux, Field *r1, Field *r2, Field *w, 
//                         Field *u, Field *v, Field *S, Qreal lambda, Qreal cn, Qreal Pe){
    
//     BwdTrans(r1->mesh,r1->spec, r1->phys);
//     BwdTrans(r2->mesh,r2->spec, r2->phys);
//     BwdTrans(w->mesh, w->spec, w->phys);
//     vel_func(w, u, v);
//     S_func(r1,r2,S);
//     int Nx = r2nonl->mesh->Nx; int Ny = r2nonl->mesh->Ny; int BSZ = r2nonl->mesh->BSZ;
//     dim3 dimGrid = r2nonl->mesh->dimGridp; dim3 dimBlock = r2nonl->mesh->dimBlockp;

//     FldMul<<<dimGrid, dimBlock>>>(S->phys, S->phys, -1.0, r2nonl->phys,Nx, Ny, BSZ);
//     FldMul<<<dimGrid, dimBlock>>>(r2->phys, r2nonl->phys, 1.0, r2nonl->phys,Nx, Ny, BSZ);
//     FwdTrans(r2nonl->mesh, r2nonl->phys, r2nonl->spec);
// }

// void Mr2nonl_func(Field* r2nonl, Field* r1curr, Field* r2curr, Field* S, Qreal t, Field *aux){
//     Mesh* mesh = r2nonl->mesh;
//     dim3 dimGrid = mesh->dimGridp;
//     dim3 dimBlock = mesh->dimBlockp;
//     BwdTrans(mesh, r1curr->spec, r1curr->phys);
//     BwdTrans(mesh, r2curr->spec, r2curr->phys);
//     // nonl = -S^2 r2
//     //cal the S
//     S_func(r1curr, r2curr, S);

//     FldMul<<<dimGrid, dimBlock>>>(S->phys, S->phys, -1.0, r2nonl->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
//     FldMul<<<dimGrid, dimBlock>>>(r2nonl->phys, r2curr->phys, 1.0, r2nonl->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

//     FwdTrans(mesh, r2nonl->phys, r2nonl->spec);
    
// }

int main(){
    int startpoint = 4720;

    int BSZ = 16;
    int Ns = 100001;
    int Nx = 512; // same as colin
    int Ny = 512;
    int Nxh = Nx/2+1;
    Qreal Lx = 4*M_PI;
    Qreal Ly = 4*M_PI;
    Qreal dx = Lx/Nx;
    Qreal dy = Ly/Ny;
    Qreal dt = 0.001; // same as colin
    // Qreal a = 1.0;

    Qreal Re = 0.1;
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
    cout << "start point = " << startpoint << endl;
    cout << "Re = " << Re << "  "; cout << "Er = " << Er << endl;
    cout << "Pe = " << Pe << "  "; cout << "Ra = " << Ra << endl;
    cout << "cf = " << cf << "  "; cout << "cn = " << cn << "  "; cout << "dt = " << dt << endl;
    cout << "Lx = " << mesh->Lx << " "<< "Ly = " << mesh->Ly << " " << endl;
    cout << "Nx = " << mesh->Nx << " "<< "Ny = " << mesh->Ny << " " << endl;
    cout << "dx = " << mesh->dx << " "<< "dy = " << mesh->dy << " " << endl;
    cout << "dt = " << dt << endl;
    cout << "Nx*dx = " << mesh->Nx*mesh->dx << " "<< "Ny*dy = " << mesh->Ny*mesh->dy << " " << endl;
    cout << "End time: Ns*dt = " << Ns*dt << endl;
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
    
    Field *u = new Field(mesh); Field *v = new Field(mesh); Field *S = new Field(mesh);
    Field *p11 = new Field(mesh); Field *p12 = new Field(mesh); Field* p21 = new Field(mesh);
    
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
    // Mr1lin_func<<<mesh->dimGridsp,mesh->dimBlocksp>>>(IFr1h, IFr1, mesh->k_squared, 
    // dt, mesh->Nxh, mesh->Ny, mesh->BSZ);
    // Mr2lin_func<<<mesh->dimGridsp,mesh->dimBlocksp>>>(IFr2h, IFr2, mesh->k_squared, 
    // dt, mesh->Nxh, mesh->Ny, mesh->BSZ);
    r1lin_func<<<mesh->dimGridsp,mesh->dimBlocksp>>>(IFr1h, IFr1, mesh->k_squared, 
    Pe, cn, dt, mesh->Nxh, mesh->Ny, mesh->BSZ);
    r2lin_func<<<mesh->dimGridsp,mesh->dimBlocksp>>>(IFr2h, IFr2, mesh->k_squared, 
    Pe, cn, dt, mesh->Nxh, mesh->Ny, mesh->BSZ);

    cuda_error_func( cudaDeviceSynchronize() );
    wlin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(IFwh, IFw, mesh->k_squared, Re, cf, dt, mesh->Nxh, mesh->Ny, mesh->BSZ);
    cuda_error_func( cudaDeviceSynchronize() );
    // the precomputation function also updates the spectrum of corresponding variables
    precompute_func(r1old, r2old, wold, Phy_init);
    alpha_init(alpha->phys, Ra, dx, dy, Nx, Ny);
    FwdTrans(mesh, alpha->phys, alpha->spec);
    
    // prepare the referenced system
    BwdTrans(mesh, wold->spec, wold->phys);
    BwdTrans(mesh, r1old->spec, r1old->phys);
    BwdTrans(mesh, r2old->spec, r2old->phys);
    S_func(r1old, r2old, S);
    cuda_error_func( cudaDeviceSynchronize() );
    
    field_visual(wold,"wstart.csv");
    field_visual(r1old,"r1start.csv");
    field_visual(r2old,"r2start.csv");
    field_visual(S,"Sstart.csv");
    coord(*mesh);
    
    //////////////////////// time stepping //////////////////////////
    for(int m=0 ;m<Ns ;m++){
        integrate_func0(r1old, r1curr, r1new, IFr1, IFr1h, dt);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func0(r2old, r2curr, r1new, IFr2, IFr2h, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        integrate_func0(wold, wcurr, wnew, IFw, IFwh, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        // Mr1nonl_func(r1nonl, r1curr, r2curr, S, m*dt, aux);
        // Mr2nonl_func(r2nonl, r1curr, r2curr, S, m*dt, aux);
        // Or1nonl_func(r1nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        r1nonl_func(r1nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        r2nonl_func(r2nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func1(r1old, r1curr, r1new, r1nonl, IFr1, IFr1h, dt);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func1(r2old, r2curr, r2new, r2nonl, IFr2, IFr2h, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        wnonl_func(wnonl, aux, aux1, p11, p12, p21, r1curr, r2curr, wcurr, u, v, alpha, S, Re, Er, cn, lambda);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func1(wold, wcurr, wnew, wnonl, IFw, IFwh, dt);

        
        // Mr1nonl_func(r1nonl, r1curr, r2curr, S, m*dt, aux);
        // Mr2nonl_func(r2nonl, r1curr, r2curr, S, m*dt, aux);
        // Or1nonl_func(r1nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        // cuda_error_func( cudaDeviceSynchronize() );
        // Or2nonl_func(r2nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        // cuda_error_func( cudaDeviceSynchronize() );
        r1nonl_func(r1nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        r2nonl_func(r2nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func2(r1old, r1curr, r1new, r1nonl, IFr1, IFr1h, dt);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func2(r2old, r2curr, r2new, r2nonl, IFr2, IFr2h, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        wnonl_func(wnonl, aux, aux1, p11, p12, p21, r1curr, r2curr, wcurr, u, v, alpha, S, Re, Er, cn, lambda);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func2(wold, wcurr, wnew, wnonl, IFw, IFwh, dt);
        
        // Mr1nonl_func(r1nonl, r1curr, r2curr, S, m*dt, aux);
        // Mr2nonl_func(r2nonl, r1curr, r2curr, S, m*dt, aux);
        // Or1nonl_func(r1nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        // cuda_error_func( cudaDeviceSynchronize() );
        // Or2nonl_func(r2nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        // cuda_error_func( cudaDeviceSynchronize() );
        r1nonl_func(r1nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        r2nonl_func(r2nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func3(r1old, r1curr, r1new, r1nonl, IFr1, IFr1h, dt);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func3(r2old, r2curr, r2new, r2nonl, IFr2, IFr2h, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        wnonl_func(wnonl, aux, aux1, p11, p12, p21, r1curr, r2curr, wcurr, u, v, alpha, S, Re, Er, cn, lambda);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func3(wold, wcurr, wnew, wnonl, IFw, IFwh, dt);
        
        // Mr1nonl_func(r1nonl, r1curr, r2curr, S, m*dt, aux);
        // Mr2nonl_func(r2nonl, r1curr, r2curr, S, m*dt, aux);
        // Or1nonl_func(r1nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        // cuda_error_func( cudaDeviceSynchronize() );
        // Or2nonl_func(r2nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        // cuda_error_func( cudaDeviceSynchronize() );
        r1nonl_func(r1nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        r2nonl_func(r2nonl, aux, r1curr, r2curr, wcurr, u, v, S, lambda, cn, Pe);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func4(r1old, r1curr, r1new, r1nonl, IFr1, IFr1h, dt);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func4(r2old, r2curr, r2new, r2nonl, IFr2, IFr2h, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        wnonl_func(wnonl, aux, aux1, p11, p12, p21, r1curr, r2curr, wcurr, u, v, alpha, S, Re, Er, cn, lambda);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func4(wold, wcurr, wnew, wnonl, IFw, IFwh, dt);
        
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1new->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
        cuda_error_func( cudaDeviceSynchronize() );
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2new->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
        cuda_error_func( cudaDeviceSynchronize() );
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wnew->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);

        if(m%10 == 0) {cout << "\r" << "t = " << (startpoint+m)*dt << flush;}
        if (m%80 == 0){
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
            // printf("t: %f    val:%.8f   exa:%.8f    err: %.8f \n",m*dt,  u->phys[5],exact((m)*dt), u->phys[5]-exact((m)*dt));
            // cout<<"t: " << m*dt << "  " << u->phys[5] << endl;
        }
        // // integrate_func0(w_old, w_curr, w_new, wIF, wIFh, dt);
        // // cuda_error_func( cudaDeviceSynchronize() );
        // integrate_func0(r1old, r1curr, r1new, r1IF, r1IFh, dt);
        // cuda_error_func( cudaDeviceSynchronize() );
        // integrate_func0(r2old, r2curr, r2new, r2IF, r2IFh, dt);
        // cuda_error_func( cudaDeviceSynchronize() );

        // // wnonl_func(wnonl, aux, aux1, p11, p12, p21, r1_curr, r2_curr, w_curr, u, v, alpha, S, Re, Er, cn, lambda);
        // // cuda_error_func( cudaDeviceSynchronize() );
        // Or1nonl_func(r1nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, cn, Pe);
        // // Mr1nonl_func(r1nonl, r1_curr, r2_curr, S, m*dt, aux);
        // cuda_error_func( cudaDeviceSynchronize() );
        // Or2nonl_func(r2nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, cn, Pe);
        // // Mr2nonl_func(r2nonl, r1_curr, r2_curr, S, m*dt, aux);
        // // cuda_error_func( cudaDeviceSynchronize() );
        // // integrate_func1(w_old, w_curr, w_new, wnonl, wIF, wIFh, dt);
        // cuda_error_func( cudaDeviceSynchronize() );
        // integrate_func1(r1_old, r1_curr, r1_new, r1nonl, r1IF, r1IFh, dt);
        // cuda_error_func( cudaDeviceSynchronize() );
        // integrate_func1(r2_old, r2_curr, r2_new, r2nonl, r2IF, r2IFh, dt);

        
        // // wnonl_func(wnonl, aux, aux1, p11, p12, p21, r1_curr, r2_curr, w_curr, u, v, alpha, S, Re, Er, cn, lambda);
        // // cuda_error_func( cudaDeviceSynchronize() );
        // Or1nonl_func(r1nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, cn, Pe);
        // // Mr1nonl_func(r1nonl, r1_curr, r2_curr, S, m*dt, aux);
        // cuda_error_func( cudaDeviceSynchronize() );
        // Or2nonl_func(r2nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, cn, Pe);
        // // Mr2nonl_func(r2nonl, r1_curr, r2_curr, S, m*dt, aux);
        // cuda_error_func( cudaDeviceSynchronize() );
        // // integrate_func2(w_old, w_curr, w_new, wnonl, wIF, wIFh, dt);
        // // cuda_error_func( cudaDeviceSynchronize() );
        // integrate_func2(r1_old, r1_curr, r1_new, r1nonl, r1IF, r1IFh, dt);
        // cuda_error_func( cudaDeviceSynchronize() );
        // integrate_func2(r2_old, r2_curr, r2_new, r2nonl, r2IF, r2IFh, dt);
        // cuda_error_func( cudaDeviceSynchronize() );

        // // wnonl_func(wnonl, aux, aux1, p11, p12, p21, r1_curr, r2_curr, w_curr, u, v, alpha, S, Re, Er, cn, lambda);
        // // cuda_error_func( cudaDeviceSynchronize() );
        // Or1nonl_func(r1nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, cn, Pe);
        // // Mr1nonl_func(r1nonl, r1_curr, r2_curr, S, m*dt, aux);
        // cuda_error_func( cudaDeviceSynchronize() );
        // Or2nonl_func(r2nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, cn, Pe);
        // // Mr2nonl_func(r2nonl, r1_curr, r2_curr, S, m*dt, aux);
        // cuda_error_func( cudaDeviceSynchronize() );
        // // integrate_func3(w_old, w_curr, w_new, wnonl, wIF, wIFh, dt);
        // // cuda_error_func( cudaDeviceSynchronize() );
        // integrate_func3(r1_old, r1_curr, r1_new, r1nonl, r1IF, r1IFh, dt);
        // cuda_error_func( cudaDeviceSynchronize() );
        // integrate_func3(r2_old, r2_curr, r2_new, r2nonl, r2IF, r2IFh, dt);
        
        // // wnonl_func(wnonl, aux, aux1, p11, p12, p21, r1_curr, r2_curr, w_curr, u, v, alpha, S, Re, Er, cn, lambda);
        // // cuda_error_func( cudaDeviceSynchronize() );
        // Or1nonl_func(r1nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, cn, Pe);
        // // Mr1nonl_func(r1nonl, r1_curr, r2_curr, S, m*dt, aux);
        // cuda_error_func( cudaDeviceSynchronize() );
        // Or2nonl_func(r2nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, cn, Pe);
        // // Mr2nonl_func(r2nonl, r1_curr, r2_curr, S, m*dt, aux);
        // cuda_error_func( cudaDeviceSynchronize() );
        // // integrate_func4(w_old, w_curr, w_new, wnonl, wIF, wIFh, dt);
        // // cuda_error_func( cudaDeviceSynchronize() );
        // integrate_func4(r1_old, r1_curr, r1_new, r1nonl, r1IF, r1IFh, dt);
        // cuda_error_func( cudaDeviceSynchronize() );
        // integrate_func4(r2_old, r2_curr, r2_new, r2nonl, r2IF, r2IFh, dt);
        // cuda_error_func( cudaDeviceSynchronize() );

        // SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(w_old->spec, w_new->spec, w_old->mesh->Nxh, w_old->mesh->Ny, w_old->mesh->BSZ);
        // cuda_error_func( cudaDeviceSynchronize() );
        // SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1_old->spec, r1_new->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
        // cuda_error_func( cudaDeviceSynchronize() );
        // SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2_old->spec, r2_new->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
        // cuda_error_func( cudaDeviceSynchronize() );

        // if (m%10 == 0) {

        // cout << "\r" << "t = " << m*dt << flush;
        // }
        // if (m%80 == 0){
        //     BwdTrans(mesh, r1_old->spec, r1_old->phys);
        //     BwdTrans(mesh, r2_old->spec, r2_old->phys);
        //     BwdTrans(mesh, w_old->spec, w_old->phys);
        //     S_func(r1_old, r2_old, S);
        //     cuda_error_func( cudaDeviceSynchronize() );

        //     field_visual(r1_old, to_string(m+startpoint)+"r1.csv");
        //     field_visual(r2_old, to_string(m+startpoint)+"r2.csv");
        //     field_visual(w_old, to_string(m+startpoint)+"w.csv");
        //     field_visual(S, to_string(m+startpoint)+"S.csv");
        //     BwdTrans(mesh, wnonl->spec, wnonl->phys);
        // BwdTrans(mesh, r1nonl->spec, r1nonl->phys);
        // BwdTrans(mesh, r2nonl->spec, r2nonl->phys);
        // cuda_error_func( cudaDeviceSynchronize() );

        // field_visual(r1nonl, to_string(m+startpoint)+"r1nonl.csv");
        // field_visual(r2nonl, to_string(m+startpoint)+"r2nonl.csv");
        // field_visual(wnonl, to_string(m+startpoint)+"wnonl.csv");
        //     if (std::isnan(r1_old->phys[0])) {cout << "NAN ";exit(0);}
        //     if (S->phys[0] > 1.5) {cout << "S >> 1.0 ";exit(0);}
        // }
    }

    delete mesh;

    return 0;
}

