// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len, float *temp) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float sum[2*BLOCK_SIZE];
  int s=1;
  if((2*blockIdx.x*blockDim.x+threadIdx.x)<len)
  {
    sum[threadIdx.x]=input[(2*blockIdx.x*blockDim.x+threadIdx.x)];
  }
  if((2*blockIdx.x*blockDim.x+threadIdx.x+blockDim.x)<len)
  {
    sum[threadIdx.x+blockDim.x] = input[(2*blockIdx.x*blockDim.x+threadIdx.x)+blockDim.x];
  }

  while(s<2*BLOCK_SIZE)
  {
    __syncthreads();
    if(((threadIdx.x+1)*s*2-1)>=0)
    {
      if(((threadIdx.x+1)*s*2-1)<2*BLOCK_SIZE)
      {
        sum[((threadIdx.x+1)*s*2-1)]+=sum[((threadIdx.x+1)*s*2-1)-s];
      }
    }
    s=s*2;
  }

  s = BLOCK_SIZE/2;
  while(s>0)
  {
    __syncthreads();
    if(((threadIdx.x+1)*s*2-1)+s<2*BLOCK_SIZE)
    {
      sum[((threadIdx.x+1)*s*2-1)+s]+=sum[((threadIdx.x+1)*s*2-1)];
    }
    s=s/2;
  }

  __syncthreads();
  if((2*blockIdx.x*blockDim.x+threadIdx.x)<len)
  {
    output[(2*blockIdx.x*blockDim.x+threadIdx.x)]=sum[threadIdx.x];
  }

  if(((2*blockIdx.x*blockDim.x+threadIdx.x)+blockDim.x)<len)
  {
    output[(2*blockIdx.x*blockDim.x+threadIdx.x)+blockDim.x] = sum[threadIdx.x+blockDim.x];
  }

  __syncthreads();
  if(threadIdx.x==blockDim.x-1)
  {
    temp[blockIdx.x]=sum[2*BLOCK_SIZE-1];
  }

}

__global__ void parallel(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float sum[2*BLOCK_SIZE];
  int s=1;
  if((2*blockIdx.x*blockDim.x+threadIdx.x)<len)
  {
    sum[threadIdx.x]=input[(2*blockIdx.x*blockDim.x+threadIdx.x)];
  }
  if((2*blockIdx.x*blockDim.x+threadIdx.x+blockDim.x)<len)
  {
    sum[threadIdx.x+blockDim.x] = input[(2*blockIdx.x*blockDim.x+threadIdx.x)+blockDim.x];
  }

  while(s<2*BLOCK_SIZE)
  {
    __syncthreads();
    if(((threadIdx.x+1)*s*2-1)>=0)
    {
      if(((threadIdx.x+1)*s*2-1)<2*BLOCK_SIZE)
      {
        sum[((threadIdx.x+1)*s*2-1)]+=sum[((threadIdx.x+1)*s*2-1)-s];
      }
    }
    s=s*2;
  }

  s = BLOCK_SIZE/2;
  while(s>0)
  {
    __syncthreads();
    if(((threadIdx.x+1)*s*2-1)+s<2*BLOCK_SIZE)
    {
      sum[((threadIdx.x+1)*s*2-1)+s]+=sum[((threadIdx.x+1)*s*2-1)];
    }
    s=s/2;
  }

  __syncthreads();
  if((2*blockIdx.x*blockDim.x+threadIdx.x)<len)
  {
    output[(2*blockIdx.x*blockDim.x+threadIdx.x)]=sum[threadIdx.x];
  }

  if(((2*blockIdx.x*blockDim.x+threadIdx.x)+blockDim.x)<len)
  {
    output[(2*blockIdx.x*blockDim.x+threadIdx.x)+blockDim.x] = sum[threadIdx.x+blockDim.x];
  }

  if(blockIdx.x>0)
  {
    output[blockIdx.x*blockDim.x+threadIdx.x] += input[blockIdx.x-1];
  }

}


__global__ void scanner(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  if(blockIdx.x>0)
  {
    output[blockIdx.x*blockDim.x+threadIdx.x] += input[blockIdx.x-1];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list
  float *array;
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  size_t t = ceil(numElements/(2 * BLOCK_SIZE * 1.0)) * sizeof(float);
  size_t val1 = ceil(numElements/(2 * BLOCK_SIZE * 1.0));
  cudaMalloc((void**)&array, t);
  //@@ Initialize the grid and block dimensions here
  dim3 b1(BLOCK_SIZE,1,1);
  dim3 g1(val1,1,1);
  dim3 b2(BLOCK_SIZE,1,1);
  dim3 g2(1,1,1);
  dim3 b3(2 * BLOCK_SIZE,1,1);
  dim3 g3(val1,1,1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<g1,b1>>>(deviceInput,deviceOutput,numElements, array);
  parallel<<<g2,b2>>>(array,array,val1);
  scanner<<<g3,b3>>>(array,deviceOutput,numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(array);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
