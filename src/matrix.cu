#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/matrix.cuh"
#include "matrix.h"

#define THR_PER_BLOCK 1024 

__global__ void matrix_mul_add_kernel(double *c, double *a, double *b, int a_rows, int a_cols, int b_cols, double *d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    double sum = 0.0;
    if (row < a_rows && col < b_cols) {
        for (int i = 0; i < a_cols; i++) {
            sum += *m_elem(a, a_cols, row, i) * *m_elem(b, b_cols, i, col);
        }
        *m_elem(c, b_cols, row, col) = sum + *m_elem(d, b_cols, row, col);
    }
}

__global__ void matrix_func_kernel(double *n, double *m, int rows, int cols, double (*func)(double)) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col < cols && row < rows) {
        *m_elem(n, cols, row, col) = func(*m_elem(m, cols, row, col));
    }
}

__global__ void matrix_mul_cnt_kernel(double *m, int rows, int cols, double cnt) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col < cols && row < rows) {
        *m_elem(m, cols, row, col) *= cnt;
    }
}

__global__ void matrix_sub_kernel(double *c, double *a, double *b, int rows, int cols){

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < cols && row < rows) {
        double sum;
        sum = *m_elem(a, cols, row, col) - *m_elem(b, cols, row, col);
        *m_elem(c, cols, row, col) = sum;
    }
}

__global__ void matrix_zero_kernel(double *m, int rows, int cols){

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < cols && row < rows) {
        *m_elem(m, cols, row, col) = 0.0;
    }
}

__global__ void matrix_mul_dot_kernel(double *c, double *a, double *b, int rows, int cols){

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < cols && row < rows) {
        double prod;
        prod = *m_elem(a, cols, row, col) * *m_elem(b, cols, row, col);
        *m_elem(c, cols, row, col) = prod;
    }
}

__global__ void matrix_transpose_kernel(double *m, double *m_t, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
        *m_elem(m_t, rows, j, i) = *m_elem(m, cols, i, j);
    }
}

__global__ void matrix_mul_kernel(double *c, double *a, double *b, int a_rows, int a_cols, int b_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < a_rows && col < b_cols) {
        double sum = 0.0;
        for (int i = 0; i < a_cols; i++) {
            sum += *m_elem(a, a_cols, row, i) * *m_elem(b, b_cols, i, col);
        }
        *m_elem(c, b_cols, row, col) = sum;
    }
}

//------------------------------------------------------------------------------------------------------------------------------//

//Combierte los indices 2D a 1D para el acceso
double m_elem(double *m, int length, int x, int y){
    return m[length * x + y];
}

// C= A * B + D
void gpu_matrix_mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double *d) {
    assert(a_cols == b_rows);
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil( (float)col / thr_per_blk );
    matrix_mul_add_kernel<<<blk_in_grid, thr_per_blk>>>(c, a, b, a_rows, a_cols, b_cols, d);
    //TODO hay que traer de vuelta los resultados de mul add
}

//Hacer uso de func a cada elem
void gpu_matrix_func(double *n, double *m, int rows, int cols, double (*func)(double)) {
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil( (float)col / thr_per_blk );
    matrix_func_kernel<<<blk_in_grid, thr_per_blk>>>(n, m, rows, cols, func);
    //TODO hay que traer de vuelta los resultados de func
}

// M * cnt
void gpu_matrix_mul_cnt(double *m, int rows, int cols, double cnt) {
    int thr_per_block = THR_PER_BLOCK;
    int blk_in_grid = ceil((float)cols / thr_per_block);
    matrix_mul_cnt_kernel<<<blk_in_grid, thr_per_block>>>(m, rows, cols, cnt);
    //TODO hay que traer de vuelta los resultados de mul cnt
}

// C = A - B
void gpu_matrix_sub(double *c, double *a, double *b, int rows, int cols){
    int thr_per_block = THR_PER_BLOCK;
    int blk_in_grid = ceil((float)cols / thr_per_block);
    matrix_sub_kernel<<<blk_in_grid, thr_per_block>>>(c, a, b, rows, cols);
    //TODO hay que traer de vuelta los resultados de sub
}

void gpu_matrix_zero(double *m, int rows, int cols){
    int thr_per_block = THR_PER_BLOCK;
    int blk_in_grid = ceil((float)cols / thr_per_block);
    matrix_zero_kernel<<<blk_in_grid, thr_per_block>>>(m, rows, cols);
    //TODO hay que traer de vuelta los resultados de sub
}

void gpu_matrix_mul_dot(double *c, double *a, double *b, int rows, int cols){
    int thr_per_block = THR_PER_BLOCK;
    int blk_in_grid = ceil((float)cols / thr_per_block);
    matrix_mul_dot_kernel<<<blk_in_grid, thr_per_block>>>(c, a, b, rows, cols);
    //TODO hay que traer de vuelta los resultados de sub
}

double *gpu_matrix_transpose(double *m, int rows, int cols) {
    double *m_t, *new_m_t;
    int i, j;

    gpuErrchk(cudaMalloc(&dev_m_t, rows * cols * sizeof(double)));

    int thr_per_block = THR_PER_BLOCK;
    int blk_in_grid = ceil((float)cols / thr_per_block);

    matrix_transpose_kernel<<<blk_in_grid, thr_per_block>>>(m, dev_m_t, rows, cols);

    gpuErrchk(cudaMemcpy(m_t, dev_m_t, rows * cols * sizeof(double), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(dev_m_t));

    return m_t;
}

void gpu_matrix_mul(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols){
    int thr_per_block = THR_PER_BLOCK;
    int blk_in_grid = ceil((float)cols / thr_per_block);
    matrix_mul_kernel<<<blk_in_grid, thr_per_block>>>(c, a, b, a_rows, a_cols, b_rows, b_cols);
}

void gpu_matrix_free(double *m){

    if (m != NULL)
        gpuErrchk(cudaFree(m));
}

void gpu_matrix_sum(double *c, double *a, double *b, int rows, int cols){
    int thr_per_block = THR_PER_BLOCK;
    int blk_in_grid = ceil((float)cols / thr_per_block);
    matrix_sum_kernel<<<blk_in_grid, thr_per_block>>>(c, a, b, rows, cols);
}
