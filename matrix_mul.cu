#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> // Required for clock()

/*
 * LANGUAGE NOTE: IS THIS A C PROGRAM?
 * -----------------------------------
 * Strictly speaking, this is **CUDA C++**.
 * 
 * 1. SYNTAX: It uses standard C syntax for the host code (main, functions, pointers).
 *    You can write standard C code here and it will work.
 * 
 * 2. EXTENSIONS: It includes special keywords (`__global__`, `<<<...>>>`) and 
 *    built-in variables (`threadIdx`, `blockIdx`) that standard C compilers 
 *    (gcc, clang, msvc) imply do not understand.
 * 
 * 3. COMPILATION: You MUST use the NVIDIA CUDA Compiler (`nvcc`) to compile this.
 *    `nvcc` separates the device code (GPU) from the host code (CPU).
 * 
 * So, it is a "C-style" program, but it is technically a C++ dialect extended for GPUs.
 */

/*
 * NAMING CONVENTIONS
 * ------------------
 * h_ : Host variable (exists on CPU RAM)
 * d_ : Device variable (exists on GPU Memory)
 * N  : Global constant (Macro)
 * 
 * WHY USE THESE?
 * Since CUDA involves moving data manually between two different memory spaces,
 * accidentally using a host pointer in a kernel or a device pointer in C++ code
 * causes immediate crashes (SegFaults). Prefixing variables helps track where
 * data lives.
 */

// Matrix dimensions (N x N)
// INCREASED SIZE to 256 to measure meaningful execution time.
// N=256 -> 256*256 = 65,536 threads.
#define N 256

// Block size (threads per block)
// CUDA GPUs execute threads in blocks. 
// We'll use 16x16 blocks (256 threads per block) which is a standard efficient size.
#define BLOCK_SIZE 16

// -------------------------------------------------------------------------
// THREAD HIERARCHY EXPLANATION
// -------------------------------------------------------------------------
// Q: Why use multiple blocks with threads, instead of 1 block with all threads?
//
// 1. SCALABILITY:
//    Real GPUs have thousands of cores. A single block runs on a single 
//    Streaming Multiprocessor (SM). If we put everything in 1 block, we limit 
//    execution to 1 SM, leaving the rest of the GPU idle.
//    By splitting work into multiple blocks, the GPU can schedule these blocks 
//    across all available SMs in parallel.
//
// 2. HARDWARE LIMITS:
//    Hardware has limits on max threads per block (typically 1024). 
//    For large matrices (e.g., N=1024 -> 1 million threads), we CANNOT fit 
//    them in one block. We MUST divide the grid into manageable blocks.
//
// 3. LOGICAL GROUPING:
//    Blocks often represent a "tile" of data. In matrix multiplication, using 
//    2D blocks (16x16 or 32x32) allows threads in the same block to share 
//    data via precise Shared Memory (L1 Cache equivalent) optimizations, 
//    which isn't possible across different blocks.
// -------------------------------------------------------------------------

// CUDA Kernel for Matrix Multiplication
//
// VISUALIZATION OF MATRIX POSITIONS
// ---------------------------------
// Matrices are stored in "Row-Major" order in memory. A 2D matrix (Row, Col) 
// is flattened into a 1D array.
// 
// Global Indexing:
//   index = Row * Width + Col
//
// Example: 2x2 Matrix A
//   A[0][0]  A[0][1]   (Row 0)
//   A[1][0]  A[1][1]   (Row 1)
//
// Memory Layout (1D Array):
//   [ A[0][0], A[0][1], A[1][0], A[1][1] ]
//   Index: 0        1        2        3
//
// Multiplication Operation (C = A * B):
// To compute one element C(row, col):
//   1. Traverse Row 'row' of Matrix A
//   2. Traverse Column 'col' of Matrix B
//   3. Multiply corresponding pairs and sum them up (Dot Product)
//
//   Matrix A Row (walking 'k' across cols)      Matrix B Col (walking 'k' down rows)
//   [ A(row, k=0), A(row, k=1) ... ]      *     [ B(k=0, col) ]
//                                               [ B(k=1, col) ]
//                                               [ ...         ]
//
__global__ void matrixMul(int *a, int *b, int *c, int n) {
    // 1. IDENTIFY THREAD POSITION
    // Each thread is responsible for ONE element in the result matrix C.
    // 'row' and 'col' define which unique element (C[row][col]) this thread computes.
    
    // row = BlockOffsetY + ThreadOffsetY
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // col = BlockOffsetX + ThreadOffsetX
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check using global coordinates
    if (row < n && col < n) {
        int temp = 0;
        
        // 2. COMPUTE DOT PRODUCT
        // We need to walk across Row 'row' of A and down Column 'col' of B.
        // 'k' is the loop variable for this walk.
        for (int k = 0; k < n; k++) {
            // A Index: row * n + k
            //   Fixed Row: 'row'
            //   Moving Column: 'k' moves 0 -> N-1
            
            // B Index: k * n + col
            //   Moving Row: 'k' moves 0 -> N-1
            //   Fixed Column: 'col'
            
            temp += a[row * n + k] * b[k * n + col];
        }
        
        // 3. STORE RESULT
        // Store in global memory at linearization of (row, col)
        c[row * n + col] = temp;

        // DEBUG PRINT - COMMENTED OUT FOR PERFORMANCE MEASUREMENT
        // Printf is very slow and will skew timing results significantly.
        //printf("[Thread (%d,%d) in Block (%d,%d)] computing C[%d][%d] (Global Index: %d)\n", 
        //       threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, row, col, row * n + col);
    }
}

// Function to initialize matrix with random numbers
void initMatrix(int *mat, int n, int val) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = val == -1 ? rand() % 10 : val;
    }
}

// Function to check result against CPU calculation
// Returns time taken in milliseconds
double verifyResult(int *a, int *b, int *c, int n) {
    int error = 0;
    clock_t start = clock(); // Start CPU timer
    
    // CPU Matrix Multiplication
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            int temp = 0;
            for (int k = 0; k < n; k++) {
                temp += a[row * n + k] * b[k * n + col];
            }
            // Verification check
            if (temp != c[row * n + col]) {
                if (error < 10) { // Print first 10 errors
                 printf("Error at (%d, %d): Host=%d, Device=%d\n", row, col, temp, c[row * n + col]);
                }
                error++;
            }
        }
    }
    
    clock_t end = clock(); // End CPU timer
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    if (error == 0) {
        printf("PASSED: Result verified successfully.\n");
    } else {
        printf("FAILED: Found %d errors.\n", error);
    }
    
    return cpu_time;
}

int main() {
    // -------------------------------------------------------------------------
    // MEMORY MANAGEMENT OVERVIEW
    // -------------------------------------------------------------------------
    // CUDA uses a heterogeneous memory model:
    // 1. Host Memory (CPU): Standard RAM, allocated with malloc/new.
    // 2. Device Memory (GPU): High-bandwidth memory on graphics card, allocated with cudaMalloc.
    //
    // Data Flow:
    // Host (Input) -> [cudaMemcpy] -> Device (Process) -> [cudaMemcpy] -> Host (Result)
    // -------------------------------------------------------------------------

    // Host pointers (CPU memory)
    int *h_a, *h_b, *h_c;
    // Device pointers (GPU memory)
    int *d_a, *d_b, *d_c;
    
    // Total size in bytes needed for one N x N matrix of integers
    size_t bytes = N * N * sizeof(int);

    // 1. Allocate memory on host (CPU)
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // 2. Initialize matrices with data
    initMatrix(h_a, N, 2); 
    initMatrix(h_b, N, 3); 

    // 3. Allocate memory on device (GPU)
    // cudaMalloc(&ptr, size)
    // - Allocates 'size' bytes of linear memory on the GPU.
    // - Returns a pointer to the allocated memory in 'ptr'.
    // - We pass the address of our pointer (&d_a) so cudaMalloc can update it.
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 4. Copy data from host to device
    // cudaMemcpy(dest, src, size, kind)
    // - Transfers data between Host and Device.
    // - Is a synchronous operation (blocks CPU until transfer is clear to start).
    // - Direction 'cudaMemcpyHostToDevice' is crucial.
    printf("Copying input data to GPU memory...\n");
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    
    // Grid calculation (Ceiling division): (N + BLOCK_SIZE - 1) / BLOCK_SIZE
    dim3 grid((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    printf("Multiplying %dx%d matrices...\n", N, N);
    
    // Start GPU timer (Host side measurement of Kernel + Sync)
    clock_t start_gpu = clock();

    // 5. Launch kernel
    // The GPU now operates on the data sitting in Device Memory (d_a, d_b).
    // It writes results into Device Memory (d_c).
    matrixMul<<<grid, threads>>>(d_a, d_b, d_c, N);
    
    // Check launch error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Sync for printf AND timing
    // Required to ensure GPU finishes before we stop the clock on CPU
    cudaDeviceSynchronize();
    
    // Stop GPU timer
    clock_t end_gpu = clock();
    double gpu_time = ((double)(end_gpu - start_gpu)) / CLOCKS_PER_SEC * 1000.0;

    // 6. Copy result from device to host
    // We bring the calculated results from GPU memory back to CPU RAM.
    // Direction 'cudaMemcpyDeviceToHost'.
    printf("Copying results back to CPU memory...\n");
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // 7. Verify and measure CPU time
    printf("Verifying result on CPU...\n");
    double cpu_time = verifyResult(h_a, h_b, h_c, N);

    // 8. Print Performance Results
    printf("\n--------------------------------------------------------------\n");
    printf("PERFORMANCE COMPARISON (Matrix Size: %dx%d)\n", N, N);
    printf("--------------------------------------------------------------\n");
    printf("GPU Execution Time: %.3f ms\n", gpu_time);
    printf("CPU Execution Time: %.3f ms\n", cpu_time);
    if (cpu_time > 0) {
        printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    }
    printf("--------------------------------------------------------------\n");

    // 9. Free memory
    // Important to prevent memory leaks on both CPU and GPU
    free(h_a); free(h_b); free(h_c); 
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
