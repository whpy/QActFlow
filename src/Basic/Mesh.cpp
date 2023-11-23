#include <Basic/Mesh.h>

Mesh::Mesh(int pBSZ, int pNx, int pNy, Qreal pLx, Qreal pLy):BSZ(pBSZ),
Nx(pNx), Ny(pNy), Lx(pLx), Ly(pLy), Nxh(pNx/2+1),dx(pLx/pNx), 
dy(pLx/pNy),alphax(2*M_PI/pLx),alphay(2*M_PI/pLy){
    cufft_error_func( cufftPlan2d( &(this->transf), Ny, Nx, CUFFT_D2Z ) );
    cufft_error_func( cufftPlan2d( &(this->inv_transf), Ny, Nx, CUFFT_Z2D ) );

    physsize = Nx*Ny*sizeof(Qreal);
    specsize = Ny*Nxh*sizeof(Qcomp);
    wavesize = Ny*Nxh*sizeof(Qreal);
    cuda_error_func( cudaMallocManaged( &(this->mphys), physsize) );
    cuda_error_func( cudaMallocManaged( &(this->mspec), specsize) );
    cuda_error_func(cudaMallocManaged( &(this->kx), sizeof(Qreal)*(Nx)));
    cuda_error_func(cudaMallocManaged( &(this->ky), sizeof(Qreal)*(Ny)));
    cuda_error_func(cudaMallocManaged( &(this->k_squared), sizeof(Qreal)*(Ny*Nxh)));
    cuda_error_func(cudaMallocManaged( &(this->cutoff), sizeof(Qreal)*(Ny*Nxh)));

    // kx initialization
    // kx[i] = alphax*[0, 1, 2, 3,...,Nxh-1]
    // alphax = 2pi/Lx
    for (int i=0; i<Nxh; i++)          
    {
        this->kx[i] = i*alphax;
    } 
    // ky initialization
    // ky[j] = alphay*[0, 1, 2, 3,...,Ny/2, -Ny/2+1, -Ny/2+2,...,-1]
    // alphay = 2pi/Ly
    for (int j=0; j<Ny; j++)          
    {
        if(j<(Ny/2+1)){
            this->ky[j] = j*alphay;
            }
        else {
            this->ky[j] = (j-Ny)*alphay;
            }
        
    } 

    // the k^2 = kx^2+ky^2
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nxh; i++){
            int c = i + j*Nxh;
            this->k_squared[c] = kx[i]*kx[i] + ky[j]*ky[j];
        }
    }

    // set up the cutoff
    int lowNy = Ny/4;
    int lowNx = Nx/4;
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nxh; i++){
            int c = i + j*Nxh;
            // set to be 1
            // -lowNy < ky <= lowNy && -lowNx < kx <= lowNx
            // (ky <= lowNy or Ny - ky > -lowNy) && (kx <= lowNx)
            if ( ( (j<= lowNy) || (Ny - j)< lowNy ) && (i<=lowNx) ){
                cutoff[c] = 1.0;
            }
            else{
                cutoff[c] = 1.0;
            }
        }
    }

    // thread information for physical space
    dimGridp = dim3(int((Nx-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
    dimBlockp = dim3(BSZ, BSZ);

    // thread information for spectral space
    dimGridsp = dim3(int((Nxh-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
    dimBlocksp = dim3(BSZ, BSZ);
}
Mesh::~Mesh(){
    cuda_error_func(cudaFree(this->mphys));
    cuda_error_func(cudaFree(this->mspec));
    cuda_error_func(cudaFree(this->kx));
    cuda_error_func(cudaFree(this->ky));
    cuda_error_func(cudaFree(this->k_squared));
    cuda_error_func(cudaFree(this->cutoff));
};