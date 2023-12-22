#include "cuda.h"
#include "kernel.hpp"
#include <iostream>

__global__ void kernel(int *a, int *b, int *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

void test_function()
{
    std::cout << "Starting CUDA" << std::endl;
    int a[1024], b[1024], c[1024];
    int *c_a, *c_b, *c_c;
    int n = 1024;
    int num_blocks = 256;

    std::cout << "Initializing arrays" << std::endl;
    cudaMalloc(&c_a, n * sizeof(int));
    cudaMalloc(&c_b, n * sizeof(int));
    cudaMalloc(&c_c, n * sizeof(int));

    std::cout << "Filling values" << std::endl;
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    std::cout << "Copying to device" << std::endl;
    cudaMemcpy(c_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Running kernel" << std::endl;
    kernel<<<num_blocks, n / num_blocks>>>(c_a, c_b, c_c, n);

    std::cout << "Copying to host" << std::endl;
    cudaMemcpy(c, c_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Printing result" << std::endl;
    for (int i = 0; i < n; i++)
    {
        std::cout << c[i] << ", ";
    }
    std::cout << std::endl;

    std::cout << "Freeing memory" << std::endl;
    cudaFree(c_a);
    cudaFree(c_b);
    cudaFree(c_c);
}
