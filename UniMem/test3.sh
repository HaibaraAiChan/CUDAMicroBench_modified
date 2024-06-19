nsys profile --trace cuda --cuda-um-cpu-page-faults true --cuda-um-gpu-page-faults true ./LowAccessDensityTest_cuda 1 134217728
nsys profile --trace cuda --sample process-tree --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true  --cudabacktrace all --cuda-memory-usage true ./LowAccessDensityTest_cuda 1 134217728
nsys nvprof --profile-api-trace=all --unified-memory-profiling=true  --print-gpu-summary  --print-gpu-trace  ./LowAccessDensityTest_cuda 1 134217728
nsys profile --stats=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --show-output=true --cuda-memory-usage=true  ./LowAccessDensityTest_cuda 1 134217728
nsys nvprof --events page_faults,page_throttled --profile-from-start off -o profile_output.nvprof ./LowAccessDensityTest_cuda 1 134217728
nsys nvprof --events page_faults,page_throttled --profile-from-start off -o profile_output.sqlite ./LowAccessDensityTest_cuda 1 134217728

sudo perf stat -e page-faults ./LowAccessDensityTest_cuda 1 134217728