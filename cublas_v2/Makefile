test0: test.cu
	nvcc test.cu -o test -lcusparse -arch=sm_70 -O3
test1: test_1024_1024_1024.cu
	nvcc test_1024_1024_1024.cu -o test_1024_1024_1024 -lcusparse -arch=sm_70 -O3
test2: test_cuDnn.cu
	nvcc test_cuDnn.cu -o test_cuDnn -lcusparse -arch=sm_70 -O3
