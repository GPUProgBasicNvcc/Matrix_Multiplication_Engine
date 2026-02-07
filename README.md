# CUDA Matrix Multiplication

A comprehensive educational CUDA C++ program that calculates $C = A \times B$. This project demonstrates explicit GPU offloading, memory management, and thread hierarchy concepts.

## 🚀 Features
- **Matrix Multiplication Kernel**: Custom CUDA kernel using global memory.
- **Detailed Comments**: Comprehensive explanations of CUDA concepts (Grids, Blocks, Threads, Memory Management).
- **Performance Analysis**: Measures and compares execution time between GPU (CUDA) and CPU.
- **Verification**: Automatically verifies GPU results against a CPU reference implementation.

## 🛠️ Prerequisites
- NVIDIA GPU
- CUDA Toolkit installed (nvcc compiler)
- Visual Studio (for Windows build environment)

## ⚡ How to Build and Run (Windows)
1.  Open a terminal (Command Prompt or PowerShell).
2.  Run the build script:
    ```cmd
    .\build.bat
    ```

## 🧠 GPU Architecture Explained

To understand *why* CUDA code is written this way, it helps to understand the hardware.

### CPU vs GPU
-   **CPU (Latency Oriented)**: Designed to finish a *single* task as fast as possible. Has few, powerful cores with large caches and complex control logic (branch prediction).
-   **GPU (Throughput Oriented)**: Designed to finish *many* tasks at the same time. Has thousands of smaller, simpler cores.

### Key Hardware Components
1.  **SM (Streaming Multiprocessor)**:
    -   Think of this as a "CPU Core" on steroids.
    -   Each SM can run multiple **Blocks** simultaneously.
    -   It contains many "CUDA Cores" (ALUs) to execute threads.
    
2.  **SIMT (Single Instruction, Multiple Threads)**:
    -   Threads don't run independently like OS threads.
    -   They run in groups called **Warps** (typically 32 threads).
    -   All threads in a warp execute the *same instruction* at the *same time* on different data.
    -   *implication*: Avoid "Thread Divergence" (e.g., if-statements where half the warp does one thing and half does another).

3.  **Memory Hierarchy**:
    -   **Global Memory**: Large but slow (like RAM). Accessible by all blocks.
    -   **Shared Memory**: Small, ultra-fast memory located *inside* the SM. Accessible only by threads in the *same block*. (Used for optimization).
    -   **Registers**: Fastest memory, private to each thread.

## 📚 Concepts Explained

### 1. CUDA C++ vs Standard C
While the code looks like C, it is actually **CUDA C++**.
-   **Host Code**: Runs on the CPU. Uses standard C syntax (`main`, `malloc`, `printf`).
-   **Device Code**: Runs on the GPU. Uses CUDA extensions like `__global__` and special variables (`threadIdx`, `blockIdx`).
-   It requires the **NVCC** (NVIDIA CUDA Compiler) to separate and compile the GPU portions.

### 2. Naming Conventions
To prevent confusion between CPU and GPU memory (which causes crashes), we use consistent prefixes:
-   **`h_`**: Host variable (lives in CPU RAM). Example: `h_a`, `h_b`.
-   **`d_`**: Device variable (lives in GPU VRAM). Example: `d_a`, `d_b`.
-   **`N`**: Global constant for matrix dimensions.

### 3. Memory Management & Data Flow
CUDA uses a **Heterogeneous Memory Model**, meaning the CPU and GPU have separate memory spaces.
1.  **Allocation**:
    -   CPU: `malloc`
    -   GPU: `cudaMalloc` (Allocates linear memory on the graphics card)
2.  **Data Transfer**:
    -   `cudaMemcpy(..., cudaMemcpyHostToDevice)`: Sends input data to GPU.
    -   `cudaMemcpy(..., cudaMemcpyDeviceToHost)`: Retrieves results from GPU.
3.  **Comparison**: This manual management is what makes it "Explicit Offloading".

### 4. Grid and Block Hierarchy
Why do we split threads into blocks?
-   **Scalability**: A single block runs on one Streaming Multiprocessor (SM). Using multiple blocks allows the GPU to spread work across all available SMs (cores).
-   **Hardware Limits**: A block typically has a limit of 1024 threads. For large matrices (e.g., N=1024 requiring 1,000,000 threads), we *must* use multiple blocks.
-   **Grid Calculation**: We calculate the number of blocks needed using a ceiling division formula: `(N + BLOCK_SIZE - 1) / BLOCK_SIZE`.

### 5. Matrix Representation (Row-Major)
Matrices in memory are 1D arrays, not 2D. We map them using "Row-Major" order.
-   **Index Formula**: `index = Row * Width + Col`
-   **Kernel Logic**:
    -   Each thread computes **one** element of result matrix $C$.
    -   It identifies its unique $(Row, Col)$ using `blockIdx` and `threadIdx`.
    -   It performs a dot product: Walking across Row A and down Column B.

### 6. Execution Timing
To measure GPU performance accurately:
-   **`clock()` on Host**: Measures wall-clock time.
-   **`cudaDeviceSynchronize()`**: Crucial! Kernel launches are *asynchronous*. The CPU returns correctly immediately after launching the kernel. We must verify synchronization to measure the actual GPU execution time.

## 📊 Performance
The program compares the time taken by the GPU kernel (+ overhead) versus a standard triple-loop CPU implementation.
-   **Small N (e.g., 64)**: CPU might be faster due to PCIe transfer overhead.
-   **Large N (e.g., 256, 1024)**: GPU becomes significantly faster (up to 100x+ speedup) as the parallel compute capability outweighs the transfer cost.
