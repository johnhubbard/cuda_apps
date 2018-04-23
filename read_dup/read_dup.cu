/*******************************************************************************
    Copyright (c) 2018 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

    Author: John Hubbard <jhubbard@nvidia.com>

*******************************************************************************/

// Purpose of this program:
//
// Run a simple CUDA program (a reduction) on one or two GPUs. Provide a way
// to compare performance between normal runs, versus runs that have
// Read Duplication set up.
//
// Note that this uses Unified Memory. However, this does not require HMM
// (Heterogeneous Memory Management, a Linux kernel 4.14+ feature), because
// it uses CUDA's special allocator: cudaMallocManaged().
//
// --John Hubbard, 22 Apr 2018

#include <stdio.h>
#include "helper_cuda.h"

#define INPUT_DATA_SIZE (4096 * 256000)

// Each thread read this many bytes from the input array:
#define STRIDE (8192)
#define THREADS (INPUT_DATA_SIZE / STRIDE)

#define THREADS_PER_BLOCK (512)
#define BLOCKS ((THREADS + THREADS_PER_BLOCK -1)/ THREADS_PER_BLOCK)

void check_answer(char *pmem, size_t size, unsigned long long gpu_answer,
                  const char *gpu_name)
{
    unsigned long long cpu_answer = 0;

    for (size_t i = 0; i < size; ++i)
    {
        cpu_answer += pmem[i] * 2;
    }

    if (cpu_answer != gpu_answer)
    {
        printf("Wrong answer on %s. Expected: %llu, actual: %llu\n",
               gpu_name, cpu_answer, gpu_answer);
        exit(1);
    }
    else
    {
        printf("PASS (%s): correct answer (sum of 2x each element, "
               "initial values: %d): %llu\n", gpu_name, pmem[0], cpu_answer);
    }
}

__global__ void hello_gpu_kernel(void)
{
    printf("GPU: hi, I am thread %d!\n", threadIdx.x);
}

// This multiplies each element in the input data (pmem) by 2, then adds that
// to a global result.
__global__ void reduction(char *pmem, size_t size, unsigned long long *result_data)
{
    const unsigned index = ((blockIdx.x * blockDim.x) + threadIdx.x) * STRIDE;
    const unsigned per_block_index = threadIdx.x;
    __shared__ unsigned long long local_result[THREADS_PER_BLOCK];
    unsigned long long block_result = 0;

    local_result[per_block_index] = 0;

    if (index + STRIDE - 1 < size)
    {
        for (int i = 0; i < STRIDE; ++i)
        {
            local_result[per_block_index] += pmem[i] * 2;
        }
    }

    __syncthreads();

    if (per_block_index == 0)
    {
        for (int i = 0; i < THREADS_PER_BLOCK; i++)
        {
            block_result += local_result[i];
        }
    }
    atomicAdd(result_data, block_result);
}

int main(int argc, char *argv[])
{
    char *input_data = NULL;
    unsigned long long *result_data_A = NULL;
    unsigned long long *result_data_B = NULL;

    cudaEvent_t start_prefetch, finish_prefetch, finish_kernel, finish_check_answer;
    float elapsed_prefetch, elapsed_kernel, elapsed_total, elapsed_cpu;

    if (argc < 2)
    {
        printf("Usage: %s firstCudaDeviceNum, secondCudaDeviceNum [use-read-dup]\n", argv[0]);
        printf("      (If a third parameter exists, then Read Duplication will be used\n");
        printf("Example:  %s 0 1 (CUDA device numbers start at zero)\n", argv[0]);
        printf("Example:  %s 1 2\n", argv[0]);
        printf("Example:  %s 0 1 read-dup\n", argv[0]);
        printf("Example:  %s 0 0 (uses a single GPU)\n", argv[0]);
        return 1;
    }

    int device_A = atoi(argv[1]);
    int device_B = atoi(argv[2]);
    bool set_read_dup = (argc == 4);

    printf("Using these CUDA devices (GPUs): %d, %d\n", device_A, device_B);

    printf("THREADS_PER_BLOCK:     %d\n", THREADS_PER_BLOCK);
    printf("INPUT_DATA_SIZE:       %d (%d pages)\n", INPUT_DATA_SIZE, INPUT_DATA_SIZE/4096);
    printf("STRIDE:                %d\n", STRIDE);
    printf("BLOCKS:                %d\n", BLOCKS);
    printf("Actual THREADS:        %d\n", BLOCKS * THREADS_PER_BLOCK);
    printf("Required THREADS:      %d\n", THREADS);
    printf("Use Read Duplication:  %d\n", set_read_dup);


    printf("Hello from the CPU!\n");
    checkCudaErrors(cudaSetDevice(device_A));
    hello_gpu_kernel<<<1,10>>>();
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMallocManaged(&input_data, INPUT_DATA_SIZE));
    checkCudaErrors(cudaMallocManaged(&result_data_A, sizeof(unsigned long long)));
    checkCudaErrors(cudaMallocManaged(&result_data_B, sizeof(unsigned long long)));

    checkCudaErrors(cudaEventCreate(&start_prefetch));
    checkCudaErrors(cudaEventCreate(&finish_prefetch));
    checkCudaErrors(cudaEventCreate(&finish_kernel));
    checkCudaErrors(cudaEventCreate(&finish_check_answer));

    checkCudaErrors(cudaEventRecord(start_prefetch, 0));

    checkCudaErrors(cudaMemPrefetchAsync(input_data, INPUT_DATA_SIZE, device_A, NULL));
    checkCudaErrors(cudaDeviceSynchronize());

    memset(input_data, 1, INPUT_DATA_SIZE);

    if (set_read_dup)
    {
        // This pre-populates the input array (which is read-only for this program)
        // on each GPU:
        checkCudaErrors(cudaMemAdvise(input_data, INPUT_DATA_SIZE, cudaMemAdviseSetReadMostly, 0));
        checkCudaErrors(cudaMemPrefetchAsync(input_data, INPUT_DATA_SIZE, device_A, NULL));
        checkCudaErrors(cudaMemPrefetchAsync(input_data, INPUT_DATA_SIZE, device_B, NULL));
    }
    else
    {
        // For the non-read-duplication case, we pre-populated the input array
        // onto one GPU, and map the other GPU to that. This will avoid thrashing,
        // and is *probably* the best performance you can get without using
        // Read Duplication. In any case, it's a standard first attempt at
        // reasonable performance.
        checkCudaErrors(cudaMemPrefetchAsync(input_data, INPUT_DATA_SIZE, device_A, NULL));
        checkCudaErrors(cudaMemAdvise(input_data, INPUT_DATA_SIZE, cudaMemAdviseSetPreferredLocation, device_A));
        checkCudaErrors(cudaMemAdvise(input_data, INPUT_DATA_SIZE, cudaMemAdviseSetAccessedBy, device_B));
    }

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(finish_prefetch, 0));

    // Run the kernel once on each GPU. Because the input data is the same for
    // each GPU, this would normally cause page thrashing. Read duplication
    // mitigates that, because the input data is read-only in this case.
    checkCudaErrors(cudaSetDevice(device_A));
    reduction<<<BLOCKS, THREADS_PER_BLOCK>>>(input_data, INPUT_DATA_SIZE, result_data_A);

    checkCudaErrors(cudaSetDevice(device_B));
    reduction<<<BLOCKS, THREADS_PER_BLOCK>>>(input_data, INPUT_DATA_SIZE, result_data_B);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaSetDevice(device_A));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaEventRecord(finish_kernel, 0));

    check_answer(input_data, INPUT_DATA_SIZE, *result_data_A, "GPU A");
    checkCudaErrors(cudaEventRecord(finish_check_answer, 0));

    check_answer(input_data, INPUT_DATA_SIZE, *result_data_B, "GPU B");

    checkCudaErrors(cudaEventSynchronize(finish_prefetch));
    checkCudaErrors(cudaEventSynchronize(finish_kernel));
    checkCudaErrors(cudaEventSynchronize(finish_check_answer));

    checkCudaErrors(cudaEventElapsedTime(&elapsed_prefetch, start_prefetch,  finish_prefetch));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_kernel,   finish_prefetch, finish_kernel));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_total,    start_prefetch,  finish_kernel));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_cpu,      finish_kernel,   finish_check_answer));

    printf("Time to prefetch data:                 %3.1f ms\n", elapsed_prefetch);
    printf("Time to run the CUDA kernel:           %3.1f ms\n", elapsed_kernel);
    printf("Time to run the CPU version:           %3.1f ms\n", elapsed_cpu);
    printf("Total GPU time: (prefetch + kernel):   %3.1f ms\n", elapsed_total);

    checkCudaErrors(cudaEventDestroy(start_prefetch));
    checkCudaErrors(cudaEventDestroy(finish_prefetch));
    checkCudaErrors(cudaEventDestroy(finish_kernel));
    checkCudaErrors(cudaEventDestroy(finish_check_answer));

    return 0;
}

