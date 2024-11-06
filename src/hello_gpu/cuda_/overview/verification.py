import numpy as np
from cuda.bindings import driver, nvrtc

from src.hello_gpu.cuda_.overview.errors import checkCudaErrors
from src.hello_gpu.cuda_.overview.kernel import saxpy

"""
In the following code example, the Driver API is initialized so that the NVIDIA driver and GPU are accessible.
Next, the GPU is queried for their compute capability.
Finally, the program is compiled to target our local compute capability architecture with FMAD enabled. The PTX
"""


def verify() -> None:
    # Initialize CUDA Driver API
    checkCudaErrors(driver.cuInit(0))

    # Retrieve handle for device 0
    cuDevice = checkCudaErrors(driver.cuDeviceGet(0))

    # Derive target architecture for device 0
    major = checkCudaErrors(driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice))
    minor = checkCudaErrors(driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice))
    arch_arg = bytes(f"--gpu-architecture=compute_{major}{minor}", "ascii")

    # Create program
    prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(str.encode(saxpy), b"saxpy.cu", 0, [], []))

    # Compile program
    opts = [b"--fmad=false", arch_arg]
    checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, 2, opts))

    # Get PTX from compilation
    ptxSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b" " * ptxSize
    checkCudaErrors(nvrtc.nvrtcGetPTX(prog, ptx))

    # Before you can use the PTX or do any work on the GPU, you must create a CUDA context.
    # CUDA contexts are analogous to host processes for the device.
    # In the following code example, a handle for compute device 0 is passed to cuCtxCreate to designate that GPU for context creation.
    #
    # Create context
    context = checkCudaErrors(driver.cuCtxCreate(0, cuDevice))

    # With a CUDA context created on device 0, load the PTX generated earlier into a module.
    # A module is analogous to dynamically loaded libraries for the device.
    # After loading into the module, extract a specific kernel with cuModuleGetFunction.
    # It is not uncommon for multiple kernels to reside in PTX.
    #
    # Load PTX as module data and retrieve function
    ptx = np.char.array(ptx)

    # Note: Incompatible --gpu-architecture would be detected here
    module = checkCudaErrors(driver.cuModuleLoadData(ptx.ctypes.data))
    kernel = checkCudaErrors(driver.cuModuleGetFunction(module, b"saxpy"))

    # Next, get all your data prepared and transferred to the GPU.
    # For increased application performance, you can input data on the device to eliminate data transfers.
    # For completeness, this example shows how you would transfer data to and from the device.
    NUM_THREADS = 512  # Threads per block
    NUM_BLOCKS = 32768  # Blocks per grid

    a = np.array([2.0], dtype=np.float32)
    n = np.array(NUM_THREADS * NUM_BLOCKS, dtype=np.uint32)
    bufferSize = n * a.itemsize

    hX = np.random.rand(n).astype(dtype=np.float32)
    hY = np.random.rand(n).astype(dtype=np.float32)
    hOut = np.zeros(n).astype(dtype=np.float32)

    # Python doesn’t have a natural concept of pointers, yet cuMemcpyHtoDAsync expects void*.
    # Therefore, XX.ctypes.data retrieves the pointer value associated with XX.
    dXclass = checkCudaErrors(driver.cuMemAlloc(bufferSize))
    dYclass = checkCudaErrors(driver.cuMemAlloc(bufferSize))
    dOutclass = checkCudaErrors(driver.cuMemAlloc(bufferSize))

    stream = checkCudaErrors(driver.cuStreamCreate(0))

    checkCudaErrors(driver.cuMemcpyHtoDAsync(dXclass, hX.ctypes.data, bufferSize, stream))
    checkCudaErrors(driver.cuMemcpyHtoDAsync(dYclass, hY.ctypes.data, bufferSize, stream))

    # With data prep and resources allocation finished, the kernel is ready to be launched.
    # To pass the location of the data on the device to the kernel execution configuration, you must retrieve the device pointer.
    # In the following code example, int(dXclass) retries the pointer value of dXclass, which is CUdeviceptr,
    # and assigns a memory size to store this value using np.array.
    #
    # Like cuMemcpyHtoDAsync, cuLaunchKernel expects void** in the argument list.
    # In the earlier code example,
    # it creates void** by grabbing the void* value of each individual argument and placing them into its own contiguous memory.
    #
    # The following code example is not intuitive
    # Subject to change in a future release
    dX = np.array([int(dXclass)], dtype=np.uint64)
    dY = np.array([int(dYclass)], dtype=np.uint64)
    dOut = np.array([int(dOutclass)], dtype=np.uint64)

    args = [a, dX, dY, dOut, n]
    args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

    checkCudaErrors(
        driver.cuLaunchKernel(
            kernel,
            NUM_BLOCKS,  # grid x dim
            1,  # grid y dim
            1,  # grid z dim
            NUM_THREADS,  # block x dim
            1,  # block y dim
            1,  # block z dim
            0,  # dynamic shared memory
            stream,  # stream
            args.ctypes.data,  # kernel arguments
            0,  # extra (ignore)
        )
    )

    checkCudaErrors(driver.cuMemcpyDtoHAsync(hOut.ctypes.data, dOutclass, bufferSize, stream))
    checkCudaErrors(driver.cuStreamSynchronize(stream))

    # Assert values are same after running kernel
    hZ = a * hX + hY
    if not np.allclose(hOut, hZ):
        raise ValueError("Error outside tolerance for host-device vectors")

    checkCudaErrors(driver.cuStreamDestroy(stream))
    checkCudaErrors(driver.cuMemFree(dXclass))
    checkCudaErrors(driver.cuMemFree(dYclass))
    checkCudaErrors(driver.cuMemFree(dOutclass))
    checkCudaErrors(driver.cuModuleUnload(module))
    checkCudaErrors(driver.cuCtxDestroy(context))

    print("All checks pass")
