nsys profile --trace cuda --cuda-um-cpu-page-faults true --cuda-um-gpu-page-faults true ./LowAccessDensityTest_cuda 1 134217728
nsys profile --trace cuda --sample process-tree --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true  --cudabacktrace all --cuda-memory-usage true ./LowAccessDensityTest_cuda 1 134217728
nsys profile --stats=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --show-output=true --cuda-memory-usage=true  ./LowAccessDensityTest_cuda 1 134217728
nsys profile --stats=true --trace=cuda --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --show-output=true --cuda-memory-usage=true  --sample=process-tree --samples-per-backtrace=4 ./test


sudo perf stat -e page-faults ./test