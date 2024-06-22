#include <cuda_runtime.h>
#include <iostream>

// Define the REAL type as float
using REAL = float;

// Kernel to perform repeated accesses to the unified memory
__global__ void UnifiedMemoryAccessKernel(REAL* x, REAL* y, int n, REAL a, int stride) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < (n / stride)) {
        for (int i = 0; i < 1000; ++i) {
            y[idx] = a * x[idx * stride];
        }
    }
}

void MemoryAccessTest(REAL* x, REAL* y, int n, REAL a, int stride) {
    REAL *d_x, *d_y;
    cudaMallocManaged(&d_x, n * sizeof(REAL));
    cudaMallocManaged(&d_y, (n / stride) * sizeof(REAL));

    // Initialize the input array on the host
    for (int i = 0; i < n; ++i) {
        d_x[i] = static_cast<REAL>(i);
    }

    // Define the number of threads and blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Force page migrations by alternating between CPU and GPU accesses
    for (int j = 0; j < 50; ++j) {
        // Access the memory on the GPU
        UnifiedMemoryAccessKernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, n, a, stride);
        cudaDeviceSynchronize();

        // Access the memory on the CPU in a way that forces page migrations
        for (int i = 0; i < n; i += stride) {
            d_x[i] += a;
            // Touching y to create more frequent page migrations
            if (i < (n / stride)) {
                y[i] = d_y[i] + a;
            }
        }

        // Synchronize to ensure all CPU accesses are done
        cudaDeviceSynchronize();
    }

    // Optionally copy the result back to the host
    cudaMemcpy(y, d_y, (n / stride) * sizeof(REAL), cudaMemcpyDeviceToHost);

    // Free the unified memory
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    const int n = 134217728; // Size of the list
    const REAL a = 2.0f; // Multiplier
    const int stride = 1; // Stride value
    REAL *x = new REAL[n];
    REAL *y = new REAL[n / stride];

    // Run the test
    MemoryAccessTest(x, y, n, a, stride);

    // Check the results
    for (int i = 0; i < 10; ++i) {
        std::cout << y[i] << " ";
    }
    std::cout << std::endl;

    delete[] x;
    delete[] y;

    return 0;
}
