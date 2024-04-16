#include <iostream>
#include <cuda.h>
#include <cublas_v2.h>

#define CHECK_CUBLAS(call)                                                  \
{                                                                           \
    const cublasStatus_t status = (call);                                    \
    if (status != CUBLAS_STATUS_SUCCESS) {                                    \
        std::cerr << "cuBLAS error occurred: " << status << std::endl; \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
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

int main() {
    const int N = 1024;
    const int K = 1024;
    const int M = 1024;

    // Allocate host memory for input matrices
    float* hostA = new float[N * K];
    float* hostB = new float[K * M];
    float* hostC = new float[N * M];

    // Initialize input matrices
    for (int i = 0; i < N * K; ++i) {
        hostA[i] = static_cast<float>(i);
    }

    for (int i = 0; i < K * M; ++i) {
        hostB[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* deviceA;
    float* deviceB;
    float* deviceC;

	GpuTimer timer0;
	GpuTimer timer1;
	GpuTimer timer2;
	timer0.Start();
    cudaMalloc((void**)&deviceA, N * K * sizeof(float));
    cudaMalloc((void**)&deviceB, K * M * sizeof(float));
    cudaMalloc((void**)&deviceC, N * M * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(deviceA, hostA, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, K * M * sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS initialization
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform matrix multiplication
	timer0.Stop();
	int n_runs = 20;
	timer1.Start();
	for (int i = 0; i < n_runs; i++) {
    	CHECK_CUBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, deviceA, N, deviceB, K, &beta, deviceC, N));
	}
	timer1.Stop();
	timer2.Start();

    // Copy output matrix from device to host
    cudaMemcpy(hostC, deviceC, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    //for (int i = 0; i < N * M; ++i) {
    //    std::cout << hostC[i] << " ";
    //}
    //std::cout << std::endl;

    // Cleanup
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    delete[] hostA;
    delete[] hostB;
    delete[] hostC;

    cublasDestroy(cublasHandle);
	timer2.Stop();
	std::cout << "Pre: " << timer0.Elapsed() << " Exec: " << timer1.Elapsed() / n_runs << " Post: " << timer2.Elapsed() << std::endl;


    return 0;
}
