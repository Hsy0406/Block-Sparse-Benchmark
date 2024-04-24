#!/bin/bash

#M=(4 8 16 32 64 128 256 512)
M=(256 512 1024 2048)
density=(0.02 0.05 0.1 0.2 0.3 0.4 0.5)

# 使用循环遍历变量 a 中的每个元素
for i in "${M[@]}"; do
	for j in "${density[@]}"; do
    	./spgemm_example_cu "$i" "$j"
	done
done
