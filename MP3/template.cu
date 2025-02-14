
#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float tile1[32][32];
  __shared__ float tile2[32][32];
  float val = 0;

  for(int i=0; i<((numAColumns-1)/32)+1; ++i)
  {
    if(((blockIdx.y*32+threadIdx.y)<numARows)&&(i*32+threadIdx.x<numAColumns))
    {
      tile1[threadIdx.y][threadIdx.x]=A[(blockIdx.y*32+threadIdx.y)*numAColumns+i*32+threadIdx.x];
    }
    else
    {
      tile1[threadIdx.y][threadIdx.x]=0;
    }
    if(((blockIdx.x*32+threadIdx.x)<numBColumns)&&(i*32+threadIdx.y<numBRows))
    {
      tile2[threadIdx.y][threadIdx.x]=B[(i*32+threadIdx.y)*numBColumns+(blockIdx.x*32+threadIdx.x)];
    }
    else
    {
      tile2[threadIdx.y][threadIdx.x]=0;
    }

    __syncthreads();

    for(int i=0; i<32; ++i)
    {
      val+=tile1[threadIdx.y][i]*tile2[i][threadIdx.x];
    }
    __syncthreads();
  }

  if((blockIdx.y*32+threadIdx.y)<numCRows&&(blockIdx.x*32+threadIdx.x)<numCColumns)
  {
    C[(blockIdx.y*32+threadIdx.y)*numCColumns+(blockIdx.x*32+threadIdx.x)]=val;
  }

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  size_t cval = numCRows*numCColumns*sizeof(float);
  hostC = (float*)malloc(cval);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  size_t aval = numARows*numAColumns*sizeof(float);
  size_t bval = numBRows*numBColumns*sizeof(float);

  cudaMalloc((void**) &deviceA, aval);
  cudaMalloc((void**) &deviceB, bval);
  cudaMalloc((void**) &deviceC, cval);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, aval, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, bval, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  size_t cc = ceil((float)numCColumns/32);
  size_t cr = ceil((float)numCRows/32);

  dim3 grid(cc,cr,1);
  dim3 block(32,32,1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<grid, block>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, cval, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
