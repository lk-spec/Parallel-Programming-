#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define widthm 3
#define widtht 3
#define widthb (widtht + widthm - 1)

//@@ Define constant memory for device kernel here
__constant__ float M[widthm][widthm][widthm];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
    __shared__ float mat[widthb][widthb][widthb];

    if ((((blockIdx.x * widtht + threadIdx.x) - 1) < x_size) &&
        (((blockIdx.y * widtht + threadIdx.y) - 1) < y_size) &&
        (((blockIdx.z * widtht + threadIdx.z) - 1) < z_size))
    {
        mat[threadIdx.z][threadIdx.y][threadIdx.x] = input[((blockIdx.z * widtht + threadIdx.z) - 1) * (y_size * x_size) + ((blockIdx.y * widtht + threadIdx.y) - 1) * (x_size) + ((blockIdx.x * widtht + threadIdx.x) - 1)];
    }
    else
    {
        mat[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (threadIdx.x < widtht && threadIdx.y < widtht && threadIdx.z < widtht && 
    (blockIdx.x * widtht + threadIdx.x) < x_size && 
    (blockIdx.y * widtht + threadIdx.y) < y_size && 
    (blockIdx.z * widtht + threadIdx.z) < z_size)
    {
        float temp = 0;
        for (int i = 0; i < widthm; i++)
        {
            for (int j = 0; j < widthm; j++)
            {
                for (int k = 0; k < widthm; k++)
                {
                    temp += M[i][j][k] * mat[threadIdx.z + i][threadIdx.y + j][threadIdx.x + k];
                }
            }
        }

        output[(blockIdx.z * widtht + threadIdx.z) * (y_size * x_size) + (blockIdx.y * widtht + threadIdx.y) * (x_size) + (blockIdx.x * widtht + threadIdx.x)] = temp;
    }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbTime_stop(GPU, "Doing GPU memory allocation");

    cudaMalloc((void **)&deviceInput, (inputLength-3) * sizeof(float));
    cudaMalloc((void **)&deviceOutput, (inputLength-3) * sizeof(float));

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbTime_stop(Copy, "Copying data to the GPU");

    cudaMemcpy(deviceInput, &hostInput[3], (inputLength-3) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M, hostKernel, widthm * widthm * widthm * sizeof(float));
  
  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
    size_t xs = ceil(((float)x_size) / widtht);
    size_t ys = ceil(((float)y_size) / widtht);
    size_t zs = ceil(((float)z_size) / widtht);
    dim3 g(xs, ys, zs);
    dim3 b(widthb, widthb, widthb);

  //@@ Launch the GPU kernel here
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");
    conv3d<<<g, b>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

 wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");

    cudaMemcpy(&hostOutput[3], deviceOutput, (inputLength-3) * sizeof(float), cudaMemcpyDeviceToHost);

 wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}