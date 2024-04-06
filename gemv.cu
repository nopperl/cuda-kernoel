#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void gemv_naive(float* M, float* v, float* out, int n, int m) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int r = i * m;
    if (i >= n) { return; }
    float dot = 0; 

    for (unsigned int k = 0; k < m; ++k) {
        dot += M[r + k] * v[k];
    }
    out[i] = dot;
}

__global__ void gemv_transpose(float* M, float* v, float* out, int n, int m) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int c = i;
    if (i >= m) { return; }
    float dot = 0; 

    for (unsigned int k = 0; k < m; ++k) {
        dot += M[c + k * n] * v[k];
    }
    out[i] = dot;
}

void dispatch_gemv(float* d_M, float* d_v, float* d_out, int n, int m) {
    const unsigned int numThreads = 256;
    unsigned int numBlocks = ceil((float)n / numThreads);
    gemv_naive<<<numBlocks, numThreads>>>(d_M, d_v, d_out, n, m);
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      exit(code);
    }
  }
}

int main() {
    const int n = 32768;  // number of rows
    const int m = 4096;  // number of columns
    size_t v_bytes = m * sizeof(float);
    const size_t M_bytes = n * m * sizeof(float);
    const size_t out_bytes = n * sizeof(float);
    float* h_v = new float[v_bytes];
    float* h_M = new float[M_bytes];
    float* h_out = new float[out_bytes];

    srand(time(0));
    // Fill h_a with random floats between 0 and 1
    for (int i = 0; i < m; ++i) {
        h_v[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    for (int i = 0; i < n * m; ++i) {
        h_M[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    float* d_v;
    float* d_M;
    float* d_out;
    cudaMalloc(&d_v, v_bytes);
    cudaMalloc(&d_M, M_bytes);
    cudaMalloc(&d_out, out_bytes);

    cudaMemcpy(d_v, h_v, v_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, M_bytes, cudaMemcpyHostToDevice);

    dispatch_gemv(d_M, d_v, d_out, n, m);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    std::cout << "Result:\n";
    for (int i = 0; i < n; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << "\n";

    delete h_v;
    delete h_M;
    delete h_out;
    cudaFree(d_v);
    cudaFree(d_M);
    cudaFree(d_out);

    return 0;
}

