/*  Kristen Guernsey
    ECE 6780
    October 14, 2024

    This program computes the dot product of two random single precision floating-point
    vectors. 
    
    Kernel 1 computes the dot product by multiplying the vectors in-place in
    shared memory, then uses parallel reduction to calculate partial sums for each 
    block. Kernel 1 sends an array of partial sums back to the CPU and the CPU adds 
    the partial sums to determine the final dot product. 
    
    Kernel 2 computes the dot product by multiplying the vectors in-place in shared
    memory, then uses parallel reduction to calculate partial sums for each block. 
    Kernel 2 uses an atomic function for read-modify-write operations involved in 
    adding up the partial sums on the GPU.

*/

#include <stdio.h>
#include <stdlib.h>

#define N 16777216
#define SHARED_MEM_AMT 1024

/* Function Declarations */
void random_floats(float *X, int vect_length);
float CPU_big_dot(float *A, float *B, int bytes);


/*  CUDA GPU kernel 1 function to calculate partial sums of the dot product of
    two vectors. Uses shared memory and parallel reduction. 
    Parameters:  
        g_A:    single precision floating point vector of length n
        g_B:    single precision floating point vector of length n
        g_ps1:  single precision floating point vector of length (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK
        n:      length of vectors g_A and g_B                   */
__global__ void dot_product_1(float *g_A, float *g_B, float *g_ps1, int n) {

    unsigned int tid = threadIdx.x;                     // thread ID
    unsigned int idx = blockIdx.x * blockDim.x + tid;   // vector index

    if (idx >= n) return;                   // boundary check

    // Save vector segments to shared memory
    __shared__ float s_A[SHARED_MEM_AMT];   // allocate shared memory
    __shared__ float s_B[SHARED_MEM_AMT];       

    float *addr_ptr_A = g_A + blockIdx.x * blockDim.x;  // find starting address of vector segment
    float *addr_ptr_B = g_B + blockIdx.x * blockDim.x;

    s_A[tid] = addr_ptr_A[tid];             // copy vector segments to shared memory
    s_B[tid] = addr_ptr_B[tid];
    __syncthreads();                        // ensure all copying to shared mem is done

    // Multiply vectors in shared memory
    s_A[tid] = s_A[tid] * s_B[tid];

    // In-place reduction in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2* stride)) == 0) {
            s_A[tid] += s_A[tid + stride];
        }
        __syncthreads();                    // synchronize within threadblock
    }

    // Partial sum for this block is stored in thread 0
    
    // Write partial sum result for this block to global memory
    if (tid == 0) g_ps1[blockIdx.x] = s_A[0];
}


/*  CUDA GPU kernel 2 function to calculate the dot product of two vectors. 
    Uses shared memory, parallel reduction and an atomic function.
    Parameters:  
        g_A:    single precision floating point vector of length n
        g_B:    single precision floating point vector of length n
        g_dp2:  single precision floating point value
        n:      length of vectors g_A and g_B                   */
__global__ void dot_product_2(float *g_A, float *g_B, float *g_dp2, int n) {

    unsigned int tid = threadIdx.x;                     // thread ID
    unsigned int idx = blockIdx.x * blockDim.x + tid;   // vector index

    if (idx >= n) return;                   // boundary check

    // Save vector segments to shared memory
    __shared__ float s_A[SHARED_MEM_AMT];   // allocate shared memory
    __shared__ float s_B[SHARED_MEM_AMT];       

    float *addr_ptr_A = g_A + blockIdx.x * blockDim.x;  // find starting address of vector segment
    float *addr_ptr_B = g_B + blockIdx.x * blockDim.x;

    s_A[tid] = addr_ptr_A[tid];             // copy vector segments to shared memory
    s_B[tid] = addr_ptr_B[tid];
    __syncthreads();                        // ensure all copying to shared mem is done

    // Multiply vectors in shared memory
    s_A[tid] = s_A[tid] * s_B[tid];

    // In-place reduction in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2* stride)) == 0) {
            s_A[tid] += s_A[tid + stride];
        }
        __syncthreads();                    // synchronize within threadblock
    }

    // Partial sum for this block is stored in thread 0

    // Add partial sum result to global dot product variable
    if (tid == 0) atomicAdd(g_dp2, s_A[0]);
}


int main(void) {

    // Host Variables
    float *A, *B;                   // host copies of vectors
    int bytes = N*sizeof(float);    // size of vecotrs in bytes
    float *ps1;                     // host copy of partial sum vector
    float dp1;                      // dot product from kernel 1
    float dp2;                      // dot product from kernel 2

    // Device Property Variables
    int device;                     // device number
    struct cudaDeviceProp prop;     // device properties
    int THREADS_PER_BLOCK;          // max threads per block

    // Device Variables
    float *d_A, *d_B;               // device copies of vectors
    float *d_ps1;                   // device copy of partial sum vector
    float *d_dp2;                   // device copy of dot product 2

    // Timers
    cudaEvent_t startCPU, stopCPU;  // timers for CPU-only
    cudaEventCreate(&startCPU);
    cudaEventCreate(&stopCPU);
    cudaEvent_t start1, stop1;      // timers for kernel 1
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEvent_t start2, stop2;      // timers for kernel 2
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

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


    // CPU-only dot product
    cudaEventRecord(startCPU);
    float CPU_dot_product = CPU_big_dot(A, B, bytes);
    cudaEventRecord(stopCPU);
    printf("\nCPU-Only Dot Product: %f \n", CPU_dot_product);
    float cpu_ms = 0;
    cudaEventElapsedTime(&cpu_ms, startCPU, stopCPU);
    printf("CPU-Only Time: %0.2f ms \n", cpu_ms);


    // CPU+GPU Dot Product (shared memory & parallel reduction)

    // Allocate Device Memory and Copy Input Vectors to Device
    cudaMalloc((void **) &d_A, bytes);
    cudaMalloc((void **) &d_B, bytes);
    cudaMalloc((void **) &d_ps1, grid.x * sizeof(float));
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    // Start Timer
    cudaDeviceSynchronize();
    cudaEventRecord(start1);

    // Kernel 1
    dot_product_1<<<grid, block>>>(d_A, d_B, d_ps1, N);

    // End Timer
    cudaDeviceSynchronize();
    cudaEventRecord(stop1);

    // Copy Partial Sums to Host
    cudaMemcpy(ps1, d_ps1, grid.x * sizeof(float), cudaMemcpyDeviceToHost);

    // Add Partial Sums and Print
    dp1 = 0;
    for (int i=0; i < grid.x; i++) {
        dp1 += ps1[i];
    }
    printf("\nKernel 1 Dot Product: %f \n", dp1);

    // Timers
    cudaEventSynchronize(stop1);
    float k1_ms = 0;
    cudaEventElapsedTime(&k1_ms, start1, stop1);

    // Print
    printf("Kernel 1 Time: %0.2f ms \n", k1_ms);


    // GPU Dot Product (adding atomic function or atomic lock)

    // Allocate Device Memory and Copy Input Vectors to Device
    cudaMalloc((void **) &d_dp2, sizeof(float));
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);
    dp2 = 0;
    cudaMemcpy(d_dp2, &dp2, sizeof(float), cudaMemcpyHostToDevice);

    // Start Timer
    cudaDeviceSynchronize();
    cudaEventRecord(start2);

    // Kernel 2
    dot_product_2<<<grid, block>>>(d_A, d_B, d_dp2, N);

    // End Timer
    cudaDeviceSynchronize();
    cudaEventRecord(stop2);

    // Copy Partial Sums to Host
    cudaMemcpy(&dp2, d_dp2, sizeof(float), cudaMemcpyDeviceToHost);

    // Timers
    cudaEventSynchronize(stop2);
    float k2_ms = 0;
    cudaEventElapsedTime(&k2_ms, start2, stop2);

    // Prints
    printf("\nKernel 2 Dot Product: %f \n", dp2);
    printf("Kernel 2 Time: %0.2f ms \n", k2_ms);


    // Clean Up
    free(A);
    free(B);
    free(ps1);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_ps1);
    cudaFree(d_dp2);

}


/* Populate a vector with floating point values
   Inputs: 
        X: single precision floating point vector
        vect_length: number of elements in vector X      */
void random_floats(float *X, int vect_length) {
    for (int i = 0; i < vect_length; i++) {
        X[i] = (float) rand() / (float) rand();
    }
    return;
}


/* Compute the dot product of two vectors on the CPU
   Input:   
        A, B:   single precision floating point vectors of length N
        bytes:   size of vector in bytes
   Output:  dot product of given vectors        */
float CPU_big_dot(float *A, float *B, int bytes) {

    float dot_product = 0.0;
    for (int i = 0; i < N; i++) {
        dot_product += (A[i] * B[i]);
    }

    return dot_product;
}