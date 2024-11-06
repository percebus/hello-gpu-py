"""
It's common practice to write CUDA kernels near the top of a translation unit, so write it next.
The entire kernel is wrapped in triple quotes to form a string.
The string is compiled later using NVRTC.
This is the only part of CUDA Python that requires some understanding of CUDA C++.
For more information, see
https://developer.nvidia.com/blog/even-easier-introduction-cuda/

Go ahead and compile the kernel into PTX.
Remember that this is executed at runtime using NVRTC.
There are three basic steps to NVRTC:
 - Create a program from the string.
 - Compile the program.
 - Extract PTX from the compiled program.
"""

saxpy = """\
extern "C" __global__
void saxpy(float a, float *x, float *y, float *out, size_t n)
{
 size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
 if (tid < n) {
   out[tid] = a * x[tid] + y[tid];
 }
}
"""
