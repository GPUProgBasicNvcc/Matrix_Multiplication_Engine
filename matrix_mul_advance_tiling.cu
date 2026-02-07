#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Matrix dimensions and Tile size
#define N 256
#define TILE_SIZE 16  // Aligned with the 16x16 thread blocks mentioned in the blog

/**
 * ARCHITECTURAL NOTE: TILED KERNEL
 * --------------------------------
 * This kernel utilizes __shared__ memory to achieve "Tiling."
 * Instead of fetching from Global Memory O(N^3) times, threads load 
 * data into the SM's local SRAM, reducing VRAM bandwidth pressure.
 */
__global__ void matrixMulTiled(int *a, int *b, int *c, int n) {
    // Shared memory allocation for the tile
    __shared__ int tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ int tile_b[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int temp = 0;

    // Iterate over tiles required to compute the result
    for (int t = 0; t < (n / TILE_SIZE); ++t) {
        
        // 1. COLLABORATIVE LOAD: Threads in a block load one tile into shared memory
        tile_a[threadIdx.y][threadIdx.x] = a[row * n + (t * TILE_SIZE + threadIdx.x)];
        tile_b[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * n + col];

        // Synchronize to ensure the entire tile is loaded before computation
        __syncthreads();

        // 2. COMPUTE: Multiply the tiles residing in high-speed SRAM
        for (int k = 0; k < TILE_SIZE; ++k) {
            temp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = temp;
    }
}

int main() {
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    size_t bytes = N * N * sizeof(int);

    // Host Allocation
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // Initialize data
    for(int i=0; i<N*N; i++) { h_a[i] = 2; h_b[i] = 3; }

    // Device Allocation
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Host to Device Transfer (DMA)
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid(N / TILE_SIZE, N / TILE_SIZE);

    // ---------------------------------------------------------
    // ENGINEERING-GRADE TIMING (CUDA EVENTS)
    // ---------------------------------------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMulTiled<<<grid, threads>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // ---------------------------------------------------------

    // Device to Host Transfer
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("Kernel Execution Time (Tiled): %.4f ms\n", milliseconds);
    printf("Speedup Analysis: Tiling reduces Global Memory transactions by a factor of %d.\n", TILE_SIZE);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    return 0;
}