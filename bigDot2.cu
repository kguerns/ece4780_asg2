/*  Kristen Guernsey
    ECE 6780
    October 14, 2024

    This program...

*/

#include <stdio.h>
#include <stdlib.h>

//#define N 1024*128
#define N 16777216

/* Function Declarations */
void random_floats(float *X, int vect_length);

__global__ void dot_product_1(float *g_A, float *g_B, float *g_ps1, int n) {

    unsigned int tid = threadIdx.x;                     // thread ID
    unsigned int idx = blockIdx.x * blockDim.x + tid;   // vector index

    if (idx >= n) return;                   // boundary check

    // Multiply vectors in-place in global memory
    g_A[idx] = g_A[idx] * g_B[idx];
    __syncthreads();                        // ensure all multiplication is done

    // Save vector segment to shared memory
    __shared__ float smem[1024];            // allocate shared memory   // TODO: change?

    float *s_A = g_A + blockIdx.x * blockDim.x;     // find starting address of vector segment

    smem[tid] = s_A[tid];                   // copy vector to shared memory
    __syncthreads();                        // ensure all copying to shared mem is done

    // In-place reduction in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2* stride)) == 0) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();                    // synchronize within threadblock
    }

    // Partial sum for this block is stored in thread 0
    
    // Write partial sum result for this block to global memory
    if (tid == 0) g_ps1[blockIdx.x] = smem[0];
}


int main(void) {

    // Host Variables
    float *A, *B;                   // host copies of vectors
    int bytes = N*sizeof(float);    // size of vecotrs in bytes
    float *ps1;                     // host copy of partial sum vector
    float dp1;                      // dot product from kernel 1

    // Device Property Variables
    int device;                     // device number
    struct cudaDeviceProp prop;     // device properties
    int THREADS_PER_BLOCK;          // max threads per block

    // Device Variables
    float *d_A, *d_B;               // device copies of vectors
    float *d_ps1;                   // device copy of partial sum vector


    // Find Device Properties
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    THREADS_PER_BLOCK = prop.maxThreadsPerBlock;

    // Set Execution Configuration
    dim3 block  (THREADS_PER_BLOCK, 1);
    dim3 grid   ((N + block.x - 1) / block.x, 1);

    // Prints
    printf("N = %d \n", N);
    printf("THREADS_PER_BLOCK = %d \n", THREADS_PER_BLOCK);
    printf("grid %d, block %d \n", grid.x, block.x);

    // Allocate Host Memory and Set Vector Values
    A = (float *) malloc(bytes);
    B = (float *) malloc(bytes);
    random_floats(A, N);
    random_floats(B, N);
    ps1 = (float *) malloc(grid.x * sizeof(float));


    // CPU+GPU Dot Product (shared memory & parallel reduction)

    // Allocate Device Memory and Copy Input Vectors to Device
    cudaMalloc((void **) &d_A, bytes);
    cudaMalloc((void **) &d_B, bytes);
    cudaMalloc((void **) &d_ps1, grid.x * sizeof(float));
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    // TODO: Start Timer

    // Kernel 1
    dot_product_1<<<grid, block>>>(d_A, d_B, d_ps1, N);
    cudaDeviceSynchronize();

    // TODO:  End Timer

    // Copy Partial Sums to Host
    cudaMemcpy(ps1, d_ps1, grid.x * sizeof(float), cudaMemcpyDeviceToHost);

    // Add Partial Sums
    dp1 = 0;
    for (int i=0; i < grid.x; i++) {
        dp1 += ps1[i];
    }
    printf("dp1 = %f \n", dp1);


    // GPU Dot Product (adding atomic function or atomic lock)



    // Clean Up
    free(A);
    free(B);
    free(ps1);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_ps1);

}




/* Populate a vector with floating point values
   Inputs: 
        X: single precision floating point vector
        vect_length: number of elements in vector X      */
void random_floats(float *X, int vect_length) {
    //srand ( time(NULL) );
    for (int i = 0; i < vect_length; i++) {
        X[i] = (float) rand() / (float) rand();
        //X[i] = 1;
    }
    return;
}