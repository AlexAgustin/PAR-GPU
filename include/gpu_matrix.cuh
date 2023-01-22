#include <stdbool.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

/* Macro for checking cuda errors following a cuda launch or api call
 Taken from: https://gist.github.com/jefflarkin/5390993 */
#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                                 \
    if(e!=cudaSuccess) {                                              \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
    exit(0); \
    }                                                                 \
}

/*#define gpuErrchk(call)                                \
    do {                                                        \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                     \
        }                                                         \
    } while (0)
*/
//---------------------------------------------------------------------------------------------------------//
double **gpu_alloc_matrix_2v(int n_layers, int *size, int *size_prev, double (*init_weight_ptr)(void));

double **gpu_alloc_matrix_1v(int n_layers, int *size, double (*init_weight_ptr)(void));

void gpu_matrix_free_2D(double **m, int n_layers);

void gpu_matrix_free(double *m);

double *gpu_m_elem(double *m, int length, int x, int y);

void gpu_matrix_sum(double *c, double *a, double *b, int rows, int cols);

void gpu_matrix_sub(double *c, double *a, double *b, int rows, int cols);

void gpu_matrix_mul_cnt(double *m, int rows, int cols, double cnt);

void gpu_matrix_zero(double *m, int rows, int cols);

void gpu_matrix_mul_dot(double *c, double *a, double *b, int rows, int cols);

double *gpu_matrix_transpose(double *m, int rows, int cols);

void gpu_matrix_mul(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols);

void gpu_matrix_mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double *d);

void gpu_matrix_func(double *n, double *m, int rows, int cols, double (*func)(double));

