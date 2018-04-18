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

    Author: John Hubbard

*******************************************************************************/

// Set up read duplication between two GPUs, and compare
// performance to a non-read-duplicated case.

#include <stdio.h>
#include "helper_cuda.h"

#define INPUT_DATA_SIZE (4096 * 256000)

// Each thread read this many bytes from the input array:
#define STRIDE (8192)
#define THREADS (INPUT_DATA_SIZE / STRIDE)

#define THREADS_PER_BLOCK (512)
#define BLOCKS ((THREADS + THREADS_PER_BLOCK -1)/ THREADS_PER_BLOCK)


void check_answer(char *pmem, size_t size, unsigned long long gpu_answer)
{
    unsigned long long cpu_answer = 0;

    for (size_t i = 0; i < size; ++i)
    {
        cpu_answer += pmem[i] * 2;
    }

    if (cpu_answer != gpu_answer)
    {
        printf("Wrong GPU answer. Expected: %llu, actual: %llu\n", cpu_answer, gpu_answer);
        exit(1);
    }
    else
    {
        printf("PASS: correct answer (sum of 2x each element, initial values: %d): %llu\n",
               pmem[0], cpu_answer);
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
    const int index = ((blockIdx.x * blockDim.x) + threadIdx.x) * STRIDE;
    unsigned long long local_result = 0;

    if (index + STRIDE - 1 < size)
    {
        for (int i = 0; i < STRIDE; ++i)
        {
            local_result += pmem[i] * 2;
        }
    }

    atomicAdd(result_data, local_result);
}

int main(int argc, char *argv[])
{
    char *input_data = NULL;
    unsigned long long *result_data = NULL;

    cudaEvent_t start_prefetch, finish_prefetch, finish_kernel, finish_check_answer;
    float elapsed_prefetch, elapsed_kernel, elapsed_total, elapsed_cpu;

    if (argc < 2)
    {
        printf("Usage: %s firstCudaDeviceNum, secondCudaDeviceNum\n", argv[0]);
        return 1;
    }

    int device_A = atoi(argv[1]);
    int device_B = atoi(argv[2]);

    printf("Using these CUDA devices (GPUs): %d, %d\n", device_A, device_B);
    checkCudaErrors(cudaSetDevice(device_A));

    printf("THREADS_PER_BLOCK: %d\n", THREADS_PER_BLOCK);
    printf("INPUT_DATA_SIZE:   %d (%d pages)\n", INPUT_DATA_SIZE, INPUT_DATA_SIZE/4096);
    printf("STRIDE:            %d\n", STRIDE);
    printf("BLOCKS:            %d\n", BLOCKS);
    printf("Actual THREADS:    %d\n", BLOCKS * THREADS_PER_BLOCK);
    printf("Required THREADS:  %d\n", THREADS);

    printf("Hello from the CPU!\n");
    hello_gpu_kernel<<<1,10>>>();
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMallocManaged(&input_data, INPUT_DATA_SIZE));
    checkCudaErrors(cudaMallocManaged(&result_data, sizeof(unsigned long long)));

    checkCudaErrors(cudaEventCreate(&start_prefetch));
    checkCudaErrors(cudaEventCreate(&finish_prefetch));
    checkCudaErrors(cudaEventCreate(&finish_kernel));
    checkCudaErrors(cudaEventCreate(&finish_check_answer));

    checkCudaErrors(cudaEventRecord(start_prefetch, 0));

    checkCudaErrors(cudaMemPrefetchAsync(input_data, INPUT_DATA_SIZE, device_A, NULL));
    checkCudaErrors(cudaDeviceSynchronize());

    memset(input_data, 1, INPUT_DATA_SIZE);
    checkCudaErrors(cudaMemAdvise(input_data, INPUT_DATA_SIZE, cudaMemAdviseSetReadMostly, 0));

    checkCudaErrors(cudaMemPrefetchAsync(input_data, INPUT_DATA_SIZE, device_A, NULL));
    checkCudaErrors(cudaMemPrefetchAsync(input_data, INPUT_DATA_SIZE, device_B, NULL));
    checkCudaErrors(cudaDeviceSynchronize());


    // Disable thrashing mitigation:
    checkCudaErrors(cudaMemAdvise(input_data, INPUT_DATA_SIZE, cudaMemAdviseSetPreferredLocation, device_A));
    checkCudaErrors(cudaMemAdvise(input_data, INPUT_DATA_SIZE, cudaMemAdviseSetAccessedBy, device_B));

    checkCudaErrors(cudaEventRecord(finish_prefetch, 0));

    reduction<<<BLOCKS, THREADS_PER_BLOCK>>>(input_data, INPUT_DATA_SIZE, result_data);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaEventRecord(finish_kernel, 0));

    check_answer(input_data, INPUT_DATA_SIZE, *result_data);
    checkCudaErrors(cudaEventRecord(finish_check_answer, 0));

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

