#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include "gpu_matrix.cuh"
#include "matrix.h"

#define THR_PER_BLOCK 1024 

__global__ void matrix_sum_kernel(double *c, double *a, double *b, int rows, int cols){

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    double sum;

    if (col < cols && row < rows) {
        //sum = *gpu_m_elem(a, cols, row, col) + *gpu_m_elem(b, cols, row, col);
        //*gpu_m_elem(c, cols, row, col) = sum;
        sum = a[cols * row + col] + b[cols * row + col];
        c[cols * row + col] = sum;
    }
}

__global__ void matrix_sub_kernel(double *c, double *a, double *b, int rows, int cols){

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    double sum;

    if (col < cols && row < rows) {
        //sum = *gpu_m_elem(a, cols, row, col) - *gpu_m_elem(b, cols, row, col);
        //*gpu_m_elem(c, cols, row, col) = sum;
        sum = a[cols * row + col] - b[cols * row + col];
        c[cols * row + col] = sum;
    }
}

__global__ void matrix_mul_cnt_kernel(double *m, int rows, int cols, double cnt) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col < cols && row < rows) {
        //*gpu_m_elem(m, cols, row, col) *= cnt;
        m[cols * row + col] *= cnt;
    }
}

__global__ void matrix_zero_kernel(double *m, int rows, int cols){

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < cols && row < rows) {
        //*gpu_m_elem(m, cols, row, col) = 0.0;
        m[cols * row + col] *= 0.0;
    }
}

__global__ void matrix_mul_dot_kernel(double *c, double *a, double *b, int rows, int cols){

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    double prod;

    if (col < cols && row < rows) {
        //prod = *gpu_m_elem(a, cols, row, col) * *gpu_m_elem(b, cols, row, col);
        //*gpu_m_elem(c, cols, row, col) = prod;
        prod = a[cols * row + col] * b[cols * row + col];
        c[cols * row + col] = prod;
    }
}

__global__ void matrix_transpose_kernel(double *m, double *m_t, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
        //*gpu_m_elem(m_t, rows, j, i) = *gpu_m_elem(m, cols, i, j);
        m_t[rows * j + i] = m[cols * i + j];
    }
}

__global__ void matrix_mul_kernel(double *c, double *a, double *b, int a_rows, int a_cols, int b_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < a_rows && col < b_cols) {
        double sum = 0.0;
        for (int i = 0; i < a_cols; i++) {
            //sum += *gpu_m_elem(a, a_cols, row, i) * *gpu_m_elem(b, b_cols, i, col);
            sum += a[a_cols * row + i] * b[b_cols * i + col];
        }
        //*gpu_m_elem(c, b_cols, row, col) = sum;
        c[b_cols * row + col] = sum;
    }
}

__global__ void matrix_mul_add_kernel(double *c, double *a, double *b, int a_rows, int a_cols, int b_cols, double *d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    double sum = 0.0;
    if (row < a_rows && col < b_cols) {
        for (int i = 0; i < a_cols; i++) {
            //sum += *gpu_m_elem(a, a_cols, row, i) * *gpu_m_elem(b, b_cols, i, col);
            sum += a[a_cols * row + i] * b[b_cols * i + col];
        }
        //*gpu_m_elem(c, b_cols, row, col) = sum + *gpu_m_elem(d, b_cols, row, col);
        c[b_cols * row + col] = sum + d[b_cols * row + col];
    }
}

__global__ void matrix_func_kernel(double *n, double *m, int rows, int cols, double (*func)(double)) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col < cols && row < rows) {
        //*gpu_m_elem(n, cols, row, col) = func(*gpu_m_elem(m, cols, row, col));
        n[cols * row + col] = func(m[cols * row + col]);
    }
}

//------------------------------------------------------------------------------------------------------------------------------//

//Reserva memoria para la matriz DE DOS DIMENSIONES tanto en el host como en el dispositivo
double **gpu_alloc_matrix_2v(int n_layers, int *size, int *size_prev, double (*init_weight_ptr)(void)) {
    double **m;
    double **d_m;
    int i, j;

    // Reservar memoria para m en el host
    if ((m = (double**)malloc(n_layers * sizeof(double*))) == NULL) {
        return(NULL);
    }

    //Reservar memoria para las layers en el host
    for (i=0; i < n_layers; i++)
        if ((m[i] = (double*)malloc(size[i] * size_prev[i] * sizeof(double))) == NULL) {
            gpu_matrix_free_2D(m, n_layers);
            return(NULL);
        }

    for (i = 0; i < n_layers; i++){
        for (j =0; j < size[i] * size_prev[i]; j++){
            m[i][j] = init_weight_ptr();
        }
    }

    //Reservar memoria para d_m en el host
    cudaMalloc(&d_m, n_layers * sizeof(double*));
    cudaCheckError();

    for (i = 0; i < n_layers; i++) {
        // Reserve memory for each layer on device
        cudaMalloc(&d_m[i], size[i] * size_prev[i] * sizeof(double));
        cudaCheckError();
    }

    for (i = 0; i < n_layers; i++) {
        for (j = 0; j < size[i] * size_prev[i]; j++) {
            cudaMemcpy(&d_m[i][j], &m[i][j], sizeof(double), cudaMemcpyHostToDevice);
            cudaCheckError();
        }
    }

    return(d_m);
}

//Reserva memoria para la matriz DE UNA DIMENSION tanto en el host como en el dispositivo
double **gpu_alloc_matrix_1v(int n_layers, int *size, double (*init_weight_ptr)(void)) {
    double **m;
    double **d_m;
    int i, j;

    // Reservar memoria para m en el host
    if ((m = (double**)malloc(n_layers * sizeof(double*))) == NULL) {
        return(NULL);
    }

    //Reservar memoria para las layers en el host
    for (i=0; i < n_layers; i++)
        if ((m[i] = (double*)malloc(size[i] * sizeof(double))) == NULL) {
            gpu_matrix_free_2D(m, n_layers);
            return(NULL);
        }

    for (i = 0; i < n_layers; i++){
        for (j =0; j < size[i]; j++){
            m[i][j] = init_weight_ptr();
        }
    }

    //Reservar memoria para d_m en el host
    cudaMalloc(&d_m, n_layers * sizeof(double*));
    cudaCheckError();

    for (i = 0; i < n_layers; i++) {
        // Reserve memory for each layer on device
        cudaMalloc(&d_m[i], size[i] * sizeof(double));
        cudaCheckError();
    }

    for (i = 0; i < n_layers; i++) {
        for (j = 0; j < size[i]; j++) {
            cudaMemcpy(&d_m[i][j], &m[i][j], sizeof(double), cudaMemcpyHostToDevice);
            cudaCheckError();
        }
    }

    return(d_m);
}

// Libera la memoria asignada a una matriz de dos dimensiones
void gpu_matrix_free_2D(double **m, int n_layers){

    int i;

    for (i=0; i < n_layers; ++i) {
        if (m[i] != NULL) {
            cudaFree(m[i]);
            cudaCheckError();
        }
    }
    cudaFree(m);
    cudaCheckError();
}

// Libera la memoria asiganda a M
void gpu_matrix_free(double *m){

    if (m != NULL)
        cudaFree(m);
        cudaCheckError();
}

//Combierte los indices 2D a 1D para el acceso
double *gpu_m_elem(double *m, int length, int x, int y){
    return (double *)&m[length * x + y];
}

// C = A + B
void gpu_matrix_sum(double *c, double *a, double *b, int rows, int cols){
    int thr_per_block = THR_PER_BLOCK;
    int blk_in_grid = ceil((float)cols / thr_per_block);
    matrix_sum_kernel<<<blk_in_grid, thr_per_block>>>(c, a, b, rows, cols);
}

// C = A - B
void gpu_matrix_sub(double *c, double *a, double *b, int rows, int cols){
    int thr_per_block = THR_PER_BLOCK;
    int blk_in_grid = ceil((float)cols / thr_per_block);
    matrix_sub_kernel<<<blk_in_grid, thr_per_block>>>(c, a, b, rows, cols);
}

// M * cnt
void gpu_matrix_mul_cnt(double *m, int rows, int cols, double cnt) {
    int thr_per_block = THR_PER_BLOCK;
    int blk_in_grid = ceil((float)cols / thr_per_block);
    matrix_mul_cnt_kernel<<<blk_in_grid, thr_per_block>>>(m, rows, cols, cnt);
}

// Cada elemento de M sera 0
void gpu_matrix_zero(double *m, int rows, int cols){
    int thr_per_block = THR_PER_BLOCK;
    int blk_in_grid = ceil((float)cols / thr_per_block);
    matrix_zero_kernel<<<blk_in_grid, thr_per_block>>>(m, rows, cols);
}

// Multiplicacion con decimal C[i] = A[i] * B[i]
void gpu_matrix_mul_dot(double *c, double *a, double *b, int rows, int cols){
    int thr_per_block = THR_PER_BLOCK;
    int blk_in_grid = ceil((float)cols / thr_per_block);
    matrix_mul_dot_kernel<<<blk_in_grid, thr_per_block>>>(c, a, b, rows, cols);
}

// Hace la traspuesta de una matriz en unna matriz auxiliar y la devuelve.
double *gpu_matrix_transpose(double *m, int rows, int cols) {
    double  *d_m_t;

    cudaMalloc(&d_m_t, rows * cols * sizeof(double));
    cudaCheckError();

    int thr_per_block = THR_PER_BLOCK;
    int blk_in_grid = ceil((float)cols / thr_per_block);

    matrix_transpose_kernel<<<blk_in_grid, thr_per_block>>>(m, d_m_t, rows, cols);

    return (d_m_t);
}

// Multiplicacion de matrices (linea * columnas)
void gpu_matrix_mul(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols){
    int thr_per_block = THR_PER_BLOCK;
    int blk_in_grid = ceil((float)a_cols / thr_per_block);
    matrix_mul_kernel<<<blk_in_grid, thr_per_block>>>(c, a, b, a_rows, a_cols, b_cols);
}

// C= A * B + D
void gpu_matrix_mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double *d) {
    assert(a_cols == b_rows);
    int thr_per_blk = THR_PER_BLOCK;
    int blk_in_grid = ceil( (float)a_cols / thr_per_blk );
    matrix_mul_add_kernel<<<blk_in_grid, thr_per_blk>>>(c, a, b, a_rows, a_cols, b_cols, d);
}

//Hacer uso de func a cada elem
void gpu_matrix_func(double *n, double *m, int rows, int cols, double (*func)(double)) {
    int thr_per_blk = THR_PER_BLOCK;
    int blk_in_grid = ceil( (float)cols / thr_per_blk );
    matrix_func_kernel<<<blk_in_grid, thr_per_blk>>>(n, m, rows, cols, func);
}