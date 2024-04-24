/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpGEMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include "dense2sparse_csr_example.cu"

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}
struct GpuTimer
{
      cudaEvent_t start;
      cudaEvent_t stop;

      GpuTimer()
      {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
      }

      ~GpuTimer()
      {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
      }

      void Start()
      {
            cudaEventRecord(start, 0);
      }

      void Stop()
      {
            cudaEventRecord(stop, 0);
      }

      float Elapsed()
      {
            float elapsed;
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            return elapsed;
      }
};

void CsrSpmm(int A_num_rows, int A_num_cols, int A_nnz, int B_num_rows, int B_num_cols,
			   	int B_nnz, int* hA_csrOffsets, int* hA_columns, float* hA_values, 
				int* hB_csrOffsets, int* hB_columns, float* hB_values) {
    // Host problem definition
    #define   A_NUM_ROWS 4   // C compatibility
    //const int A_num_rows = 4;
    //const int A_num_cols = 4;
    //const int A_nnz      = 9;
    //const int B_num_rows = 4;
    //const int B_num_cols = 4;
    //const int B_nnz      = 9;
    //int   hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
    //int   hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    //float hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
    //                          6.0f, 7.0f, 8.0f, 9.0f };
    //int   hB_csrOffsets[] = { 0, 2, 4, 7, 8 };
    //int   hB_columns[]    = { 0, 3, 1, 3, 0, 1, 2, 1 };
    //float hB_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
    //                          6.0f, 7.0f, 8.0f };
    const int C_nnz       = 12;
    #define   C_NUM_NNZ 12   // C compatibility
    float               alpha       = 1.0f;
    float               beta        = 0.0f;
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_32F;
    //--------------------------------------------------------------------------
    // Device memory management: Allocate and copy A, B
    int   *dA_csrOffsets, *dA_columns, *dB_csrOffsets, *dB_columns,
          *dC_csrOffsets, *dC_columns;
    float *dA_values, *dB_values, *dC_values;
    // allocate A
    cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int));
    cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))  ;
    cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float));
    // allocate B
    cudaMalloc((void**) &dB_csrOffsets,
                           (B_num_rows + 1) * sizeof(int));
    cudaMalloc((void**) &dB_columns, B_nnz * sizeof(int))  ;
    cudaMalloc((void**) &dB_values,  B_nnz * sizeof(float));
    // allocate C offsets
    cudaMalloc((void**) &dC_csrOffsets,
                           (A_num_rows + 1) * sizeof(int));

    // copy A
    cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dA_values, hA_values,
                           A_nnz * sizeof(float), cudaMemcpyHostToDevice);
    // copy B
    cudaMemcpy(dB_csrOffsets, hB_csrOffsets,
                           (B_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dB_columns, hB_columns, B_nnz * sizeof(int),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dB_values, hB_values,
                           B_nnz * sizeof(float), cudaMemcpyHostToDevice);
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    cusparseCreate(&handle);
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_nnz,
                                      dB_csrOffsets, dB_columns, dB_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                                      dC_csrOffsets, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseSpGEMM_createDescr(&spgemmDesc);

    // ask bufferSize1 bytes for external memory
    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL);
    cudaMalloc((void**) &dBuffer1, bufferSize1);
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1);

    // ask bufferSize2 bytes for external memory
    cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, matA, matB, &beta, matC,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL);
    cudaMalloc((void**) &dBuffer2, bufferSize2);

    // compute the intermediate product of A * B
	GpuTimer timer;
	int n_runs = 20;
	timer.Start();
	for (int i = 0; i < n_runs; i++) {
    	cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, matA, matB, &beta, matC,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, dBuffer2);
	}
	timer.Stop(); 
	std::cout << " Tensor shape: " << B_num_rows << "*" << B_num_cols; 
	std::cout << " Time: " << timer.Elapsed() / n_runs << " ms" << std::endl;
    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                         &C_nnz1);
    // allocate matrix C
    cudaMalloc((void**) &dC_columns, C_nnz1 * sizeof(int))  ;
    cudaMalloc((void**) &dC_values,  C_nnz1 * sizeof(float));

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of dC_values, and before the call of cusparseSpGEMM_copy

    // update matC with the new pointers
    cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values);

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

    // copy the final products to the matrix C
    cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);

    // destroy matrix/vector descriptors
    cusparseSpGEMM_destroyDescr(spgemmDesc);
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);
    cusparseDestroy(handle);
    //--------------------------------------------------------------------------
    // device result check
    int   hC_csrOffsets_tmp[A_NUM_ROWS + 1];
    int   hC_columns_tmp[C_NUM_NNZ];
    float hC_values_tmp[C_NUM_NNZ];
    cudaMemcpy(hC_csrOffsets_tmp, dC_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyDeviceToHost);
    cudaMemcpy(hC_columns_tmp, dC_columns, C_nnz * sizeof(int),
                           cudaMemcpyDeviceToHost);
    cudaMemcpy(hC_values_tmp, dC_values, C_nnz * sizeof(float),
                           cudaMemcpyDeviceToHost);
    //--------------------------------------------------------------------------
    // device memory deallocation
    cudaFree(dBuffer1);
    cudaFree(dBuffer2);
    cudaFree(dA_csrOffsets);
    cudaFree(dA_columns);
    cudaFree(dA_values);
    cudaFree(dB_csrOffsets);
    cudaFree(dB_columns);
    cudaFree(dB_values);
    cudaFree(dC_csrOffsets);
    cudaFree(dC_columns);
    cudaFree(dC_values);
}

void test(int block_size, float density) {
	//dense2csr()
    //int A_num_rows = 4;
    //int A_num_cols = 4;
    //int A_nnz      = 9;
    //int B_num_rows = 4;
    //int B_num_cols = 4;
    //int B_nnz      = 9;
    //int   hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
    //int   hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    //float hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
    //                          6.0f, 7.0f, 8.0f, 9.0f };
    //int   hB_csrOffsets[] = { 0, 2, 4, 7, 8 };
    //int   hB_columns[]    = { 0, 3, 1, 3, 0, 1, 2, 1 };
    //float hB_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
    //                          6.0f, 7.0f, 8.0f };
	int    A_num_rows = block_size;
	int    A_num_cols = block_size;
	float  A_density = 1.0f;
	int*   A_csr_offsets;
	int*   A_csr_columns;
	float* A_csr_values;
	
	int    B_num_rows = block_size;
	int    B_num_cols = block_size;
	float  B_density = density;
	int*   B_csr_offsets;
	int*   B_csr_columns;
	float* B_csr_values;

	int A_nnz = (int) (A_num_rows * A_num_cols * A_density);
	A_csr_offsets = (int*)malloc((A_num_rows+1)*sizeof(int));
	A_csr_columns = (int*)malloc(A_nnz*sizeof(int));
	A_csr_values  = (float*)malloc(A_nnz*sizeof(float));
	
	int B_nnz = (int) (B_num_rows * B_num_cols * B_density);
	B_csr_offsets = (int*)malloc((B_num_rows+1)*sizeof(int));
	B_csr_columns = (int*)malloc(B_nnz*sizeof(int));
	B_csr_values  = (float*)malloc(B_nnz*sizeof(float));

	dense2csr(A_num_rows, A_num_cols, A_density, A_csr_offsets, A_csr_columns, A_csr_values);
	dense2csr(B_num_rows, B_num_cols, B_density, B_csr_offsets, B_csr_columns, B_csr_values);
	std::cout << "Density: " << B_density;
	CsrSpmm(A_num_rows, A_num_cols, A_nnz, B_num_rows, B_num_cols,
			   	B_nnz, A_csr_offsets, A_csr_columns, A_csr_values, 
				B_csr_offsets, B_csr_columns, B_csr_values);
	//free(A_csr_offsets);
	//free(A_csr_columns);
	//free(A_csr_values);
	//free(B_csr_offsets);
	//free(B_csr_columns);
	//free(B_csr_values);
}

int main(int argc, char* argv[]) {
	int M = (int) (std::stoi(argv[1]));
	float density = (float) (std::strtof(argv[2], nullptr));
	test(M, density);
	return 0;

}
