#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// simple function to maximize the tile width according to the the problem width and available resources
int getOptimalTileWidth(int problemWidth, int warpSize, int maxThreadsPerBlock, size_t maxSharedMemoryPerBlock) {
    // calculate the optimal tile width according to max threads per block and shared memory size
    int optimalTileWidth = sqrt(min((float)problemWidth / (float)maxThreadsPerBlock, (float)maxSharedMemoryPerBlock / (float)sizeof(float)));

    // make sure the tile width is a multiple of the warp size
    if (optimalTileWidth % warpSize != 0) {
        optimalTileWidth -= optimalTileWidth % warpSize;
    }

    // limit the tile width to the maximum threads per block
    if (optimalTileWidth > (int)sqrt(maxThreadsPerBlock)) {
        optimalTileWidth = 2;
        while (optimalTileWidth * 2 <= sqrt(maxThreadsPerBlock)) {
            optimalTileWidth *= 2;
        }
    }

    return optimalTileWidth;
}


template<int tw>  // sets the tile width, the shared memory size is assumed to be tw * tw * 2 * sizeof(float)
__global__ void tiled_matrix_multiplication_math(float *m, float *n, float *out, int h, int w, int k) {
    int tc = threadIdx.x;  // tile column worked on by this thread
    int tr = threadIdx.y;  // tile row worked on by this thread
    int c = blockIdx.x * blockDim.x + tc;  // global column index worked on by this thread
    int r = blockIdx.y * blockDim.y + tr;  // global row index worked on by this thread
    bool height_guard = r < h;
    bool width_guard = c < w;
    // declares pointer to shared memory of the size specified at kernel launch
    extern __shared__ float ms[];
    float *ns = &ms[tw * tw];  // shared memory size is assumed to be 2 * tw * tw, each input matrix takes up half
    // stores result of the dot product of the entire row and column worked on by this thread
    float p = 0.0f;
    // Split the input row and column into tiles and iteratively add the tile row and column dot product to the total
    for (int ph = 0; ph < ceil(k / (float) tw); ++ph) {
        int idx = ph * tw;  // offset index for the current tile in the global matrix
        bool inner_col_guard = idx + tc < k;
        bool inner_row_guard = idx + tr < k;
        // fill shared memory with the input values for the current tile
        // each thread writes one value for each input matrix
        ms[tr * tw + tc] = height_guard && inner_col_guard ? m[idx + tc + r * k] : 0.0f;
        ns[tr * tw + tc] = width_guard && inner_row_guard ? n[(idx + tr) * w + c] : 0.0f;
        // wait until all threads in the block have finished filling the shared memory
        __syncthreads();
        // calculate dot product of tile row and column and update total dot product
        for (int i = 0; i < tw; ++i) {
            p += ms[tr * tw + i] * ns[i * tw + tc];
        }
        // wait until all threads in the block have finished updating their dot product
        __syncthreads();
    }
    if (height_guard && width_guard) {
        out[r * w + c] = p;
    }
}

torch::Tensor tiled_matrix_multiplication(torch::Tensor m, torch::Tensor n) {
    int h = m.size(0);
    int k = m.size(1);
    int w = n.size(1);
    TORCH_CHECK(k == n.size(0));
    auto output = torch::empty({h, w}, m.options());
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int TW = getOptimalTileWidth(w * h, prop.warpSize, prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
    size_t shared_size = TW * TW * 2 * sizeof(float);
    dim3 threads_per_block(TW, TW);
    dim3 blocks(ceil(w / (float)threads_per_block.x), ceil(h / (float)threads_per_block.y));
    auto f = [&](auto kernel_f) { kernel_f<<<blocks, threads_per_block, shared_size>>>(
        m.data_ptr<float>(), n.data_ptr<float>(), output.data_ptr<float>(), h, w, k
    );};
    // provides option to choose different tile sizes. Since the static shared memory
    // requires the size to be known at compile time, only a predetermined number
    // of possible values are provided using templates
    switch (TW) {
        case 8: f(tiled_matrix_multiplication_math<8>); break;
        case 16: f(tiled_matrix_multiplication_math<16>); break;
        case 32: f(tiled_matrix_multiplication_math<32>); break;
        case 64: f(tiled_matrix_multiplication_math<64>); break;
        case 96: f(tiled_matrix_multiplication_math<96>); break;
        case 128: f(tiled_matrix_multiplication_math<128>); break;
        default: break;
    }
    return output;
}
