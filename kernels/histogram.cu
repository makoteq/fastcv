#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

struct apply_mask {
    int* masked_count;

    apply_mask(int* count) : masked_count(count) {}

    __host__ __device__
    unsigned char operator()(const thrust::tuple<unsigned char, unsigned char>& T) const {
        unsigned char pixel = thrust::get<0>(T);
        unsigned char mask_val = thrust::get<1>(T);
        if (mask_val == 0) {
            // required to compile
             #if defined(__CUDA_ARCH__)
                atomicAdd(masked_count, 1);
            #endif
            return 0;
        }
        return pixel;
    }
};

__global__ void calcHistKernel(const unsigned char* in, int* out, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ int temp_hist[256];

    int total_threads_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;
    for (int i = total_threads_in_block; i < 256; i += block_size) {
        temp_hist[i] = 0;
    }
    
    __syncthreads();

    if (col < w && row < h) {
        int idx = row * w + col;
        unsigned char val = in[idx];
        atomicAdd(&temp_hist[val], 1);
    }

    __syncthreads();

    for (int i = total_threads_in_block; i < 256; i += block_size) {
        if (temp_hist[i] > 0) {
            atomicAdd(&out[i], temp_hist[i]);
        }
    }
    __syncthreads();
}

std::vector<float> calcHist(torch::Tensor img, c10::optional<torch::Tensor> mask) {
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);
    
    const int height = img.size(0);
    const int width = img.size(1);
    const int pixels = width * height;

    int range[2] = {0, 256}; 
    
    torch::Tensor work_img;

    bool mask_flag = mask.has_value() && mask.value().defined() && mask.value().numel() > 0;

    if (mask_flag) {
        work_img = img.clone(); 
    } else {
        work_img = img; 
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    thrust::device_ptr<unsigned char> img_pointer(work_img.data_ptr<unsigned char>());
    
    int* device_masked_count = nullptr;
    int host_masked_count = 0;

    if (mask_flag) {
        cudaMalloc(&device_masked_count, sizeof(int));
        cudaMemsetAsync(device_masked_count, 0, sizeof(int), stream);

        thrust::device_ptr<unsigned char> mask_pointer(mask.value().data_ptr<unsigned char>());
        
        auto first = thrust::make_zip_iterator(thrust::make_tuple(img_pointer, mask_pointer));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(img_pointer + pixels, mask_pointer + pixels));

        thrust::transform(thrust::cuda::par.on(stream), first, last, img_pointer, apply_mask(device_masked_count));
    }

    int hist_size = range[1];

    int* hist;
    cudaMalloc(&hist, hist_size * sizeof(int));

    cudaMemsetAsync(hist, 0, hist_size * sizeof(int), stream);

    dim3 dimBlock = getOptimalBlockDim(width, height);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    calcHistKernel<<<dimGrid, dimBlock, 0, stream>>>(
        work_img.data_ptr<unsigned char>(),
        hist,
        width, height
    );
    
    std::vector<int> host_hist(hist_size);
    cudaMemcpyAsync(host_hist.data(), hist, hist_size * sizeof(int), cudaMemcpyDeviceToHost, stream);

    if (mask_flag) {
        cudaMemcpyAsync(&host_masked_count, device_masked_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    }

    cudaStreamSynchronize(stream);

    if (mask_flag) {
        host_hist[0] -= host_masked_count;
        cudaFree(device_masked_count);
    }

    cudaFree(hist);
    cudaStreamDestroy(stream);

    std::vector<float> result(hist_size);
    for (int i = 0; i < hist_size; i++) {
        result[i] = static_cast<float>(host_hist[i]);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}