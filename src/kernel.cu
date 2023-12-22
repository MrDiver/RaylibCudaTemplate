#include "cuda.h"
#include "kernel.hpp"
#include <iostream>
#include <stdio.h>

__global__ void kernel(int *a, int *b, int *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void set_zero(float *a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        a[i] = 0;
    }
}

__global__ void init_triangles(Triangle *triangles, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        triangles[i].a = i * 3;
        triangles[i].b = i * 3 + 1;
        triangles[i].c = i * 3 + 2;
    }
}

GPUMesh *create_mesh(uint32_t vertex_count, float *vertices)
{
    GPUMesh *mesh;
    cudaMallocManaged(&mesh, sizeof(GPUMesh));
    cudaMalloc(&mesh->vertices, vertex_count * 3 * sizeof(float));
    cudaMalloc(&mesh->vertex_offsets, vertex_count * 3 * sizeof(float));
    cudaMalloc(&mesh->triangles, (vertex_count / 3) * sizeof(Triangle));
    mesh->vertex_count = vertex_count;
    mesh->triangle_count = vertex_count / 3;

    cudaMemcpy(mesh->vertices, vertices, vertex_count * 3 * sizeof(float), cudaMemcpyHostToDevice);
    int num_blocks = 1024;
    cudaMemset(mesh->vertex_offsets, 0, vertex_count * 3 * sizeof(float));
    init_triangles<<<num_blocks, (vertex_count / 3) / num_blocks>>>(mesh->triangles, vertex_count / 3);

    return mesh;
}

void destroy_mesh(GPUMesh *mesh)
{
    cudaFree(mesh->vertices);
    cudaFree(mesh->triangles);
    cudaFree(mesh);
}

__global__ void offsets_to_vertices_kernel(GPUMesh *mesh, float *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < mesh->vertex_count)
    {
        out[i * 3] = mesh->vertices[i * 3] + mesh->vertex_offsets[i * 3];
        out[i * 3 + 1] = mesh->vertices[i * 3 + 1] + mesh->vertex_offsets[i * 3 + 1];
        out[i * 3 + 2] = mesh->vertices[i * 3 + 2] + mesh->vertex_offsets[i * 3 + 2];
        // printf("%f, %f, %f\n", mesh->vertex_offsets[i * 3], mesh->vertex_offsets[i * 3 + 1],
        //        mesh->vertex_offsets[i * 3 + 2]);
    }
}

void from_gpu(GPUMesh *mesh, Mesh *out_mesh)
{
    out_mesh->vertexCount = mesh->vertex_count;
    out_mesh->triangleCount = mesh->triangle_count;
    out_mesh->vertices = (float *)malloc(mesh->vertex_count * 3 * sizeof(float));
    float *c_out_vertices;
    cudaMalloc(&c_out_vertices, mesh->vertex_count * 3 * sizeof(float));
    int num_blocks = 1024;
    offsets_to_vertices_kernel<<<num_blocks, mesh->vertex_count / num_blocks>>>(mesh, c_out_vertices);
    cudaMemcpy(out_mesh->vertices, c_out_vertices, mesh->vertex_count * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    // printf("%f %f %f\n", out_mesh->vertices[0], out_mesh->vertices[1], out_mesh->vertices[2]);
    cudaFree(c_out_vertices);
}

GPUMesh *to_gpu(Mesh *mesh)
{
    GPUMesh *m = create_mesh(mesh->vertexCount, mesh->vertices);
    return m;
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

__device__ V3 v3_add(V3 a, V3 b)
{
    V3 c = {a.x + b.x, a.y + b.y, a.z + b.z};
    return c;
}

__device__ V3 v3_sub(V3 a, V3 b)
{
    V3 c = {a.x - b.x, a.y - b.y, a.z - b.z};
    return c;
}

__device__ V3 v3_cross(V3 a, V3 b)
{
    V3 c = {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    return c;
}

__device__ V3 v3_normalize(V3 x)
{
    float length = sqrt(x.x * x.x + x.y * x.y + x.z * x.z);
    V3 c = {x.x / length, x.y / length, x.z / length};
    return c;
}

__device__ V3 calculate_normal(Triangle t, float *vertices)
{
    V3 a = {vertices[t.a * 3], vertices[t.a * 3 + 1], vertices[t.a * 3 + 2]};
    V3 b = {vertices[t.b * 3], vertices[t.b * 3 + 1], vertices[t.b * 3 + 2]};
    V3 c = {vertices[t.c * 3], vertices[t.c * 3 + 1], vertices[t.c * 3 + 2]};
    V3 ab = v3_sub(b, a);
    V3 ac = v3_sub(c, a);
    V3 normal = v3_cross(ab, ac);
    return v3_normalize(normal);
}

__global__ void wobble_kernel(GPUMesh *mesh, float time, float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < mesh->vertex_count)
    {
        float x = mesh->vertices[i * 3];
        V3 normal = calculate_normal(mesh->triangles[i / 3], mesh->vertices);
        float off = sin(time + x) * scale;
        mesh->vertex_offsets[i * 3] = off * normal.x / 10;
        mesh->vertex_offsets[i * 3 + 1] = off * normal.y / 10;
        mesh->vertex_offsets[i * 3 + 2] = off * normal.z / 10;
    }
}

void bunny_test_wobble(GPUMesh *mesh, float time, float scale)
{
    int num_blocks = 1024;
    wobble_kernel<<<num_blocks, mesh->vertex_count / num_blocks>>>(mesh, time, scale);
}
