nsys profile --trace cuda --cuda-um-cpu-page-faults true --cuda-um-gpu-page-faults true ./LowAccessDensityTest_cuda 1 134217728
nsys profile --trace cuda --sample process-tree --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true  --cudabacktrace all --cuda-memory-usage true ./LowAccessDensityTest_cuda 1 134217728
nsys profile --stats=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --show-output=true --cuda-memory-usage=true  ./LowAccessDensityTest_cuda 1 134217728
nsys profile --stats=true --trace=cuda --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --show-output=true --cuda-memory-usage=true  --sample=process-tree --samples-per-backtrace=4
 ./modified_LowAccessDensityTest_cuda 1 134217728


nsys nvprof --profile-from-start off -o profile_output.nvprof ./LowAccessDensityTest_cuda 1 134217728
nsys nvprof --profile-from-start=off -o profile_output.sqlite ./LowAccessDensityTest_cuda 1 134217728
nsys nvprof --profile-api-trace=all   ./LowAccessDensityTest_cuda 1 134217728


sudo perf stat -e page-faults ./LowAccessDensityTest_cuda 1 134217728