#pragma once
#include "raylib.h"
#include <stdint.h>
struct V3
{
    float x;
    float y;
    float z;
} typedef V3;

struct Triangle
{
    uint32_t a;
    uint32_t b;
    uint32_t c;
} typedef Triangle;

struct GPUMesh
{
    uint32_t vertex_count;
    uint32_t triangle_count;
    float *vertices;
    float *vertex_offsets;
    Triangle *triangles;
} typedef GPUMesh;

GPUMesh *create_mesh(uint32_t vertex_count, float *vertices);
void destroy_mesh(GPUMesh *mesh);

void from_gpu(GPUMesh *mesh, Mesh *out_mesh);
GPUMesh *to_gpu(Mesh *mesh);

void test_function();
void bunny_test_wobble(GPUMesh *mesh, float time, float scale);
