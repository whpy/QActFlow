#include "MQsolver.cuh"

int main(){
    // computation parameters
    int BSZ = 16;
    int Ns = 200000;
    int Nx = 768; // same as colin
    int Ny = 768;
    int Nxh = Nx/2+1;
    Qreal Lx = 33 * 2 *M_PI;
    Qreal Ly = Lx;
    Qreal dx = Lx/Nx;
    Qreal dy = dx;
    Qreal dt = 0.0001; // same as colin

    // non-dimensional number
    Qreal Re = 0.1;
    Qreal Er = 0.1;
    Qreal Rf = 7.5 * 0.00001;
    Qreal lambda = 0.1;
    Qreal ra = 0.2;

    // main Fields to be solved
    // *_curr act as an intermediate while RK4 timeintegration
    // *_new store the value of next time step 

    Mesh *mesh = new Mesh(BSZ, Nx, Ny, Lx, Ly);
    cout << "Re = " << Re << endl;
    cout << "Er = " << Er << endl;
    cout << "lambda = " << lambda << endl;
    cout << "Rf = " << Rf << endl; cout << "Ra = " << ra << endl;
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

    //////////////////////// precomputation //////////////////////////
    r1lin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1IFh, r1IF, dt, r1_old->mesh->Nxh, r1_old->mesh->Ny, r1_old->mesh->BSZ);
    r2lin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2IFh, r2IF, dt, r2_old->mesh->Nxh, r2_old->mesh->Ny, r2_old->mesh->BSZ);
    wlin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wIFh, wIF, w_old->mesh->k_squared, Re, Rf, dt, w_old->mesh->Nxh, w_old->mesh->Ny, w_old->mesh->BSZ);

    // the precomputation function also updates the spectrum of corresponding variables
    precompute_func(r1_old, r2_old, w_old, Phy_init);
    Ra_init(Ra->phys, ra, dx, dy, Nx, Ny);
    FwdTrans(mesh, Ra->phys, Ra->spec);
    cuda_error_func( cudaDeviceSynchronize() );
    coord(*mesh);
    field_visual(w_old, "wstart.csv");
    field_visual(r1_old, "r1start.csv");
    field_visual(r2_old, "r2start.csv");

    for(int m=0 ;m<Ns ;m++){
        integrate_func0(w_old, w_curr, w_new, wIF, wIFh, dt);
        integrate_func0(r1_old, r1_curr, r1_new, r1IF, r1IFh, dt);
        integrate_func0(r2_old, r2_curr, r2_new, r2IF, r2IFh, dt);
        // dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(r1_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        // dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(r2_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        // dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(w_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        
        curr_func(r1_curr, r2_curr, w_curr, u, v, S, h11, h12);
        wnonl_func(wnonl, aux, aux1, p11, p12, p21, 
        r1_curr, r2_curr, w_curr, u, v, Ra, S, Re, Er, lambda, aux, aux1);
        r1nonl_func(r1nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, aux, aux1);
        r2nonl_func(r2nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, aux, aux1);
        integrate_func1(w_old, w_curr, w_new, wnonl, wIF, wIFh, dt);
        integrate_func1(r1_old, r1_curr, r1_new, r1nonl, r1IF, r1IFh, dt);
        integrate_func1(r2_old, r2_curr, r2_new, r2nonl, r2IF, r2IFh, dt);
       
        // dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(r1_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        // dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(r2_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        // dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(w_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        curr_func(r1_curr, r2_curr, w_curr, u, v, S, h11, h12);
        wnonl_func(wnonl, aux, aux1, p11, p12, p21, 
        r1_curr, r2_curr, w_curr, u, v, Ra, S, Re, Er, lambda, aux, aux1);
        r1nonl_func(r1nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, aux, aux1);
        r2nonl_func(r2nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, aux, aux1);
        integrate_func2(w_old, w_curr, w_new, wnonl, wIF, wIFh, dt);
        integrate_func2(r1_old, r1_curr, r1_new, r1nonl, r1IF, r1IFh, dt);
        integrate_func2(r2_old, r2_curr, r2_new, r2nonl, r2IF, r2IFh, dt);
        
        // dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(r1_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        // dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(r2_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        // dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(w_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        curr_func(r1_curr, r2_curr, w_curr, u, v, S, h11, h12);
        wnonl_func(wnonl, aux, aux1, p11, p12, p21, 
        r1_curr, r2_curr, w_curr, u, v, Ra, S, Re, Er, lambda, aux, aux1);
        r1nonl_func(r1nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, aux, aux1);
        r2nonl_func(r2nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, aux, aux1);
        integrate_func3(w_old, w_curr, w_new, wnonl, wIF, wIFh, dt);
        integrate_func3(r1_old, r1_curr, r1_new, r1nonl, r1IF, r1IFh, dt);
        integrate_func3(r2_old, r2_curr, r2_new, r2nonl, r2IF, r2IFh, dt);
        
        // dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(r1_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        // dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(r2_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        // dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(w_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        curr_func(r1_curr, r2_curr, w_curr, u, v, S, h11, h12);
        wnonl_func(wnonl, aux, aux1, p11, p12, p21, 
        r1_curr, r2_curr, w_curr, u, v, Ra, S, Re, Er, lambda, aux, aux1);
        r1nonl_func(r1nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, aux, aux1);
        r2nonl_func(r2nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, aux, aux1);
        integrate_func4(w_old, w_curr, w_new, wnonl, wIF, wIFh, dt);
        integrate_func4(r1_old, r1_curr, r1_new, r1nonl, r1IF, r1IFh, dt);
        integrate_func4(r2_old, r2_curr, r2_new, r2nonl, r2IF, r2IFh, dt);

        cuda_error_func( cudaDeviceSynchronize() );
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(w_old->spec, w_new->spec, 
        w_old->mesh->Nxh, w_old->mesh->Ny, w_old->mesh->BSZ);

        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1_old->spec, r1_new->spec, 
        r1_old->mesh->Nxh, r1_old->mesh->Ny, r1_old->mesh->BSZ);

        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2_old->spec, r2_new->spec, 
        r2_old->mesh->Nxh, r2_old->mesh->Ny, r2_old->mesh->BSZ);
        dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(r1_old->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(r2_old->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(w_old->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);

        if (m%10 == 0) {

        cout << "\r" << "t = " << m*dt << flush;
        }
        if (m%200 == 0){
            BwdTrans(mesh, r1_old->spec, r1_old->phys);
            BwdTrans(mesh, r2_old->spec, r2_old->phys);
            BwdTrans(mesh, w_old->spec, w_old->phys);
            S_func(r1_old, r2_old, S);
            cuda_error_func( cudaDeviceSynchronize() );

            field_visual(r1_old, to_string(m)+"r1.csv");
            field_visual(r2_old, to_string(m)+"r2.csv");
            field_visual(w_old, to_string(m)+"w.csv");
            field_visual(S, to_string(m)+"S.csv");
            if (std::isnan(r1_old->phys[0])) {"NAN ";exit(0);}
        }
}
}

