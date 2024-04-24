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
#include <cusparse.h>         // cusparseSparseToDense
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <random>
#include <iostream>

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

float get_random()
{
	static std::default_random_engine e;
	static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
	float a = dis(e);
	if (a == 0.0) {
		return 0.05f;
	}
	else {
		return dis(e);
	}
}
void dense2csr(int num_rows, int num_cols, float density, int* h_csr_offsets, int* h_csr_columns, float* h_csr_values) {
    int   ld         = num_cols;
    int   dense_size = ld * num_rows;
	float* h_dense;
	h_dense = (float*)malloc(dense_size*sizeof(float));
    //--------------------------------------------------------------------------
	//input initialization 
	int dense_num = (int)(density * dense_size);
	int h_nnz = 0;
	std::vector<float> out0;
    for (int i = 0; i < dense_size; ++i) {
		float rate = get_random();
		if (h_nnz < dense_num){
			if ((rate <= density) || (h_nnz < (int) (dense_num * 0.9f))) {
				h_dense[i] = rate;
				h_nnz += 1;
				out0.push_back(rate);
			}
		}
		else {
			h_dense[i] = 0.0f;
		}
    }
	if (dense_num != out0.size()) {
		std::cout << "the size is wrong!" << std::endl;	
	}
    //--------------------------------------------------------------------------
    // Device memory management
    int   *d_csr_offsets, *d_csr_columns;
    float *d_csr_values,  *d_dense;

    cudaMalloc(&d_dense, dense_size * sizeof(float));
    cudaMalloc(&d_csr_offsets, (num_rows + 1) * sizeof(int));
    cudaMemcpy(d_dense, h_dense, dense_size * sizeof(float), cudaMemcpyHostToDevice);
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matA;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    cusparseCreate(&handle);
    // Create dense matrix A
    cusparseCreateDnMat(&matA, num_rows, num_cols, ld, d_dense,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW);
    // Create sparse matrix B in CSR format
    cusparseCreateCsr(&matB, num_rows, num_cols, 0,
                                      d_csr_offsets, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // allocate an external buffer if needed
    cusparseDenseToSparse_bufferSize( handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // execute Sparse to Dense conversion
    cusparseDenseToSparse_analysis(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer);
    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp,
                                         &nnz);

    // allocate CSR column indices and values
	nnz = out0.size();
    cudaMalloc((void**) &d_csr_columns, nnz * sizeof(int));
    cudaMalloc((void**) &d_csr_values,  nnz * sizeof(float));
    // reset offsets, column indices, and values pointers
    cusparseCsrSetPointers(matB, d_csr_offsets, d_csr_columns,
                                           d_csr_values);
    // execute Sparse to Dense conversion
    cusparseDenseToSparse_convert(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer);
    // destroy matrix/vector descriptors
    cusparseDestroyDnMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroy(handle);
    //--------------------------------------------------------------------------
    // device result check
    cudaMemcpy(h_csr_offsets, d_csr_offsets,
                           (num_rows + 1) * sizeof(int),
                           cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csr_columns, d_csr_columns, nnz * sizeof(int),
                           cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csr_values, d_csr_values, nnz * sizeof(float),
                           cudaMemcpyDeviceToHost);
	//for (int i=0; i< out0.size(); i++) {
	//	if (out0[i] != h_csr_values0[i]) {
	//		std::cout << "Find a wrong number, out0: " << out0[i] << " h_csr_values: " << h_csr_values[i] << " i:" << i << std::endl;
	//	}
	//}
	//std::cout << std::endl;
    //--------------------------------------------------------------------------
    // device memory deallocation
    cudaFree(dBuffer);
    cudaFree(d_csr_offsets);
    cudaFree(d_csr_columns);
    cudaFree(d_csr_values);
    cudaFree(d_dense);
	free(h_dense);
}
//int main(void) {
//	int    num_rows = 64;
//	int    num_cols = 64;
//	float  density = 0.5f;
//	int*   out_csr_offsets;
//	int*   out_csr_columns;
//	float* out_csr_values;
//
//	int nnz = (int) (num_rows * num_cols * density);
//	out_csr_offsets = (int*)malloc((num_rows+1)*sizeof(int));
//	out_csr_columns = (int*)malloc(nnz*sizeof(int));
//	out_csr_values  = (float*)malloc(nnz*sizeof(float));
//
//	dense2csr(num_rows, num_cols, density, out_csr_offsets, out_csr_columns, out_csr_values);
//	std::cout << "out_csr_offsets: ";
//   	for (int i=0; i<num_rows+1; i++) {
//		std::cout << out_csr_offsets[i] << " ";
//	}	
//	std::cout << std::endl;
//	std::cout << "out_csr_columns: ";
//   	for (int i=0; i<nnz; i++) {
//		std::cout << out_csr_columns[i] << " ";
//	}	
//	std::cout << std::endl;
//	std::cout << "out_csr_values: ";
//   	for (int i=0; i<nnz; i++) {
//		std::cout << out_csr_values[i] << " ";
//	}	
//	std::cout << std::endl;
//	dense2csr(); //num_rows, num_cols, density);
//    return 0;
//}
