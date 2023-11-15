#include "Qsolver.cuh"
using std::string;
using std::cout; 
using std::endl;



// void r1_init(Qreal *r1, Qreal dx, Qreal dy, int Nx, int Ny){
//     for (int j=0; j<Ny; j++){
//         for (int i=0; i<Nx; i++){
//             int index = i+j*Nx;
//             float x = dx*i;
//             float y = dy*j;
//             r1[index] = (float(rand())/RAND_MAX)-0.5;
//         }
//     }
// }

// void r2_init(Qreal *r2, Qreal dx, Qreal dy, int Nx, int Ny){
//     for (int j=0; j<Ny; j++){
//         for (int i=0; i<Nx; i++){
//             int index = i+j*Nx;
//             float x = dx*i;
//             float y = dy*j;
//             r2[index] = (float(rand())/RAND_MAX)-0.5;
//         }
//     }
// }


// void w_init(Qreal *w, Qreal dx, Qreal dy, int Nx, int Ny){
//     for (int j=0; j<Ny; j++){
//         for (int i=0; i<Nx; i++){
//             int index = i+j*Nx;
//             w[index] = 0.000;
//         }
//     }
// }

// void alpha_init(Qreal *alpha, Qreal Ra, Qreal dx, Qreal dy, int Nx, int Ny){
//     for (int j=0; j<Ny; j++){
//         for (int i=0; i<Nx; i++){
//             int index = i+j*Nx;
//             alpha[index] = (Ra);
//         }
//     }
// }
// void precompute_func(Field* r1, Field* r2, Field* w, InitType flag){
//     Mesh* mesh = r1->mesh;
//     int Nx = mesh->Nx; int Ny = mesh->Ny;
//     Qreal dx = mesh->dx; Qreal dy = mesh->dy;

//     if (flag == Def_init){
//         r1_init(r1->phys, dx, dy, Nx, Ny);
//         r2_init(r2->phys, dx, dy, Nx, Ny);
//         w_init(w->phys, dx, dy, Nx, Ny);
//     }
//     else if (flag == File_init){
//         file_init("./init/r1_init.csv", r1);
//         file_init("./init/r2_init.csv", r2);
//         file_init("./init/w_init.csv", w);
//     }

//     FwdTrans(mesh, r1->phys, r1->spec);
//     FwdTrans(mesh, r2->phys, r2->spec);
//     FwdTrans(mesh, w->phys, w->spec);
// }


int main(){
    int startpoint = 19980;
    // computation parameters
    int BSZ = 16;
    int Ns = 40000;
    int Nx = 512; // same as colin
    int Ny = 512;
    int Nxh = Nx/2+1;
    Qreal Lx = 33 * 2 *M_PI;
    Qreal Ly = Lx;
    Qreal dx = Lx/Nx;
    Qreal dy = dx;
    Qreal dt = 0.001; // same as colin
    // Qreal a = 1.0;

    //////////////////////// variables definitions //////////////////////////

    // non-dimensional number in "The role of advective inertia in active nematic"  
    double Re_n = 0.1;
    // to distinguish the Er we use in theis programe
    // we denotes the Er in paper as Er_n
    double Er_n = 0.1;
    // Ra = (ln/la)^2, which the square of \tiled(alpha), noted that it assumes to be negative
    // and we assume to be positive 
    double Ra = -0.2; 
    // Rf = (ln/lf)^2, which we defines cf = lc/lf where lc = ln in prl.
    // so cf = sqrt(Rf)
    double Rf = 7.5*0.00001;
    Qreal lambda = 0.1;
    
    // C_{cn} = lc/ln = 1.0 as colin set lc = ln.
    // Qreal cn = lc/ln;
    Qreal cn = 1.0;
    // C_{cf} = lc/lf = ln/lf = sqrt(Rf)
    // Qreal cf = lc/lf;
    Qreal cf = sqrt(Rf);
    Qreal Er = Er_n;
    Qreal Re = Re_n;

    // by specially choosing the scales, colin makes the Pe equals to 1.0
    Qreal Pe = 1.0;
    
    // main Fields to be solved
    // *_curr act as an intermediate while RK4 timeintegration
    // *_new store the value of next time step 

    Mesh *mesh = new Mesh(BSZ, Nx, Ny, Lx, Ly);
    cout << "start point: " << startpoint << endl;
    cout << "Re = " << Re << endl;
    cout << "Er = " << Er << endl;
    cout << "Ra = " << Ra << endl;
    cout << "Pe = " << Pe << endl;
    cout << "cf = " << cf << endl;
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
    Field *wnonl = new Field(mesh); Field *r1nonl = new Field(mesh); Field *r2nonl = new Field(mesh);
    Field *u = new Field(mesh); Field *v = new Field(mesh); Field *S = new Field(mesh);
    Field *p11 = new Field(mesh); Field *p12 = new Field(mesh); Field* p21 = new Field(mesh);
    
    // auxiliary fields
    Field *aux = new Field(mesh); Field *aux1 = new Field(mesh); 

    // field \alpha to be modified (scalar at the very first)
    Field *alpha = new Field(mesh);

    //////////////////////// precomputation //////////////////////////
    r1lin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1IFh, r1IF, r1_old->mesh->k_squared, Re, cn, dt, r1_old->mesh->Nxh, r1_old->mesh->Ny, r1_old->mesh->BSZ);
    r2lin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2IFh, r2IF, r2_old->mesh->k_squared, Re, cn, dt, r2_old->mesh->Nxh, r2_old->mesh->Ny, r2_old->mesh->BSZ);
    wlin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wIFh, wIF, w_old->mesh->k_squared, Re, cf, dt, w_old->mesh->Nxh, w_old->mesh->Ny, w_old->mesh->BSZ);
    
    // the precomputation function also updates the spectrum of corresponding variables
    precompute_func(r1_old, r2_old, w_old, File_init);
    alpha_init(alpha->phys, Ra, dx, dy, Nx, Ny);
    FwdTrans(mesh, alpha->phys, alpha->spec);
    // prepare the referenced system
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(w_old,"wstart.csv");
    field_visual(r1_old,"r1start.csv");
    field_visual(r2_old,"r2start.csv");
    coord(*mesh);
    
    //////////////////////// time stepping //////////////////////////
    for(int m=0 ;m<Ns ;m++){
        // cout << "flag 6" << endl;
        curr_func(r1_curr, r2_curr, w_curr, u, v, S);
        integrate_func0(w_old, w_curr, w_new, wIF, wIFh, dt);
        integrate_func0(r1_old, r1_curr, r1_new, r1IF, r1IFh, dt);
        integrate_func0(r2_old, r2_curr, r2_new, r2IF, r2IFh, dt);
        // cout << "flag 7" << endl;
        // cuda_error_func( cudaDeviceSynchronize() );
        // BwdTrans(mesh, ucurr->spec, ucurr->phys);
        dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(r1_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(r2_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(w_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        curr_func(r1_curr, r2_curr, w_curr, u, v, S);
        // cout << "flag 9" << endl;
        wnonl_func(wnonl, aux, aux1, p11, p12, p21, r1_curr, r2_curr, w_curr, u, v, alpha, S, Re, Er, cn, lambda);
        // cout  << "flag 11" << endl;
        r1nonl_func(r1nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, cn, Pe);
        r2nonl_func(r2nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, cn, Pe);
        // cout  << "flag 10" << endl;
        integrate_func1(w_old, w_curr, w_new, wnonl, wIF, wIFh, dt);
        integrate_func1(r1_old, r1_curr, r1_new, r1nonl, r1IF, r1IFh, dt);
        integrate_func1(r2_old, r2_curr, r2_new, r2nonl, r2IF, r2IFh, dt);
        // cout << "flag 8" << endl;
        // cuda_error_func( cudaDeviceSynchronize() );
        // unonl_func(unonl, ucurr, m*dt);
        // cuda_error_func( cudaDeviceSynchronize() );
        // integrate_func1(u, ucurr, unew, unonl, IFu, IFuh, dt);
        // BwdTrans(mesh, ucurr->spec, ucurr->phys);
        dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(r1_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(r2_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(w_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        curr_func(r1_curr, r2_curr, w_curr, u, v, S);
        wnonl_func(wnonl, aux, aux1, p11, p12, p21, r1_curr, r2_curr, w_curr, u, v, alpha, S, Re, Er, cn, lambda);
        r1nonl_func(r1nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, cn, Pe);
        r2nonl_func(r2nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, cn, Pe);
        integrate_func2(w_old, w_curr, w_new, wnonl, wIF, wIFh, dt);
        integrate_func2(r1_old, r1_curr, r1_new, r1nonl, r1IF, r1IFh, dt);
        integrate_func2(r2_old, r2_curr, r2_new, r2nonl, r2IF, r2IFh, dt);
        
        // cuda_error_func( cudaDeviceSynchronize() );
        // unonl_func(unonl, ucurr, m*dt);
        // integrate_func2(u, ucurr, unew, unonl, IFu, IFuh, dt);
        // BwdTrans(mesh, ucurr->spec, ucurr->phys);
        dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(r1_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(r2_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(w_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        curr_func(r1_curr, r2_curr, w_curr, u, v, S);
        wnonl_func(wnonl, aux, aux1, p11, p12, p21, r1_curr, r2_curr, w_curr, u, v, alpha, S, Re, Er, cn, lambda);
        r1nonl_func(r1nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, cn, Pe);
        r2nonl_func(r2nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, cn, Pe);
        integrate_func3(w_old, w_curr, w_new, wnonl, wIF, wIFh, dt);
        integrate_func3(r1_old, r1_curr, r1_new, r1nonl, r1IF, r1IFh, dt);
        integrate_func3(r2_old, r2_curr, r2_new, r2nonl, r2IF, r2IFh, dt);
        
        // cuda_error_func( cudaDeviceSynchronize() );
        // unonl_func(unonl, ucurr, m*dt);
        // integrate_func3(u, ucurr, unew, unonl, IFu, IFuh, dt);
        // BwdTrans(mesh, ucurr->spec, ucurr->phys);
        dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(r1_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(r2_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        dealiasing_func<<<mesh->dimBlocksp, mesh->dimBlocksp>>>(w_curr->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
        curr_func(r1_curr, r2_curr, w_curr, u, v, S);
        wnonl_func(wnonl, aux, aux1, p11, p12, p21, r1_curr, r2_curr, w_curr, u, v, alpha, S, Re, Er, cn, lambda);
        r1nonl_func(r1nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, cn, Pe);
        r2nonl_func(r2nonl, aux, r1_curr, r2_curr, w_curr, u, v, S, lambda, cn, Pe);
        integrate_func4(w_old, w_curr, w_new, wnonl, wIF, wIFh, dt);
        integrate_func4(r1_old, r1_curr, r1_new, r1nonl, r1IF, r1IFh, dt);
        integrate_func4(r2_old, r2_curr, r2_new, r2nonl, r2IF, r2IFh, dt);
        curr_func(r1_curr, r2_curr, w_curr, u, v, S);
        // cuda_error_func( cudaDeviceSynchronize() );
        // unonl_func(unonl, ucurr, m*dt);
        // integrate_func4(u, ucurr, unew, unonl, IFu, IFuh, dt);
        // BwdTrans(mesh, ucurr->spec, ucurr->phys);
        // cuda_error_func( cudaDeviceSynchronize() );
        // unonl_func(unonl, ucurr, m*dt);
        // cout << "flag 8" << endl;
        cuda_error_func( cudaDeviceSynchronize() );
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(w_old->spec, w_new->spec, w_old->mesh->Nxh, w_old->mesh->Ny, w_old->mesh->BSZ);
        // cout << "flag 13" << endl;
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1_old->spec, r1_new->spec, r1_old->mesh->Nxh, r1_old->mesh->Ny, r1_old->mesh->BSZ);
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2_old->spec, r2_new->spec, r2_old->mesh->Nxh, r2_old->mesh->Ny, r2_old->mesh->BSZ);
        // cout << "flag 12" << endl;
        // SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(u->spec, unew->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
        // cuda_error_func( cudaDeviceSynchronize() );

        if (m%10 == 0) {

        cout << "\r" << "t = " << m*dt << flush;
        }
        if (m%80 == 0){
            BwdTrans(mesh, r1_old->spec, r1_old->phys);
            BwdTrans(mesh, r2_old->spec, r2_old->phys);
            BwdTrans(mesh, w_old->spec, w_old->phys);
            cuda_error_func( cudaDeviceSynchronize() );

            field_visual(r1_old, to_string(m+startpoint)+"r1.csv");
            field_visual(r2_old, to_string(m+startpoint)+"r2.csv");
            field_visual(w_old, to_string(m+startpoint)+"w.csv");
            if (std::isnan(r1_old->phys[0])) {"NAN ";exit(0);}
        }
        // if(m%100 == 0) cout << "t = " << m*dt << endl;
        // if (m%200 == 0){
        //     BwdTrans(mesh, u->spec, u->phys);
        //     cuda_error_func( cudaDeviceSynchronize() );
        //     field_visual(u, to_string(m)+"u.csv");
        //     // printf("t: %f    val:%.8f   exa:%.8f    err: %.8f \n",m*dt,  u->phys[5],exact((m)*dt), u->phys[5]-exact((m)*dt));
        //     // cout<<"t: " << m*dt << "  " << u->phys[5] << endl;
        // }
    }

    return 0;
}

