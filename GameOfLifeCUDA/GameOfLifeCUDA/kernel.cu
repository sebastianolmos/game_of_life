
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h> 

typedef unsigned char ubyte;
typedef unsigned short ushort;
typedef unsigned int uint;

void initBuffers(ubyte*& data, size_t worldWidth, size_t worldHeight) {
    size_t dataLength = worldWidth * worldHeight;

    for (size_t i = 0; i < dataLength; i++) {
        data[i] = rand() & 1;
    }
}


__global__ void gameKernel(const ubyte* data, ubyte* resultData, uint worldWidth, uint worldHeight)
{
    uint worldSize = worldWidth * worldHeight;
    
    uint cell = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    for (cell; cell < worldSize; cell += blockDim.x * gridDim.x) {
        uint x = cell % worldWidth;
        uint y = cell - x;
        uint xLeft = (x + worldWidth - 1) % worldWidth;
        uint xRight = (x + 1) % worldWidth;
        uint yUp = (y + worldSize - worldWidth) % worldSize;
        uint yDown = (y + worldWidth) % worldSize;

        uint aliveCells = data[xLeft + yUp] + data[x + yUp] + data[xRight + yUp] + data[xLeft + y]
            + data[xRight + y] + data[xLeft + yDown] + data[x + yDown] + data[xRight + yDown];

        resultData[x + y] = aliveCells == 3 || (aliveCells == 2 && data[x + y]) ? 1 : 0;
    }
}

double runGameKernel(ubyte*& d_data, ubyte*& d_resultData, size_t worldWidth,
    size_t worldHeight, size_t iterations, ushort threads) {

    size_t calcBlocks = (worldWidth * worldHeight) / threads;
    ushort blocks = (ushort)std::min((size_t)32768, calcBlocks);
    cudaEvent_t start, stop;
    double totalTime = 0.0;

    for (size_t i = 0; i < iterations; i++) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        gameKernel <<<blocks, threads >>> (d_data, d_resultData, worldWidth, worldHeight);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        double gpuTime = (double)milliseconds / 1000;
        totalTime += gpuTime;

        std::swap(d_data, d_resultData);
    }
    return totalTime;
}

int main(int argc, char* argv[])
{   
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    ubyte* h_data;
    ubyte* h_resultData;

    size_t worldHeight;
    size_t worldWidth;
    size_t dataLength;
    size_t iterations;
    ushort threads;

    if (argc < 5)
    {
        worldWidth = 100;
        worldHeight = 100;
        iterations = 10;
        threads = 32;
    }
    else {
        worldWidth = atoi(argv[1]);
        worldHeight = atoi(argv[2]);
        iterations = atoi(argv[3]);
        threads = atoi(argv[4]);
    }

    dataLength = worldWidth * worldHeight;
    size_t size = dataLength * sizeof(ubyte);

    // Pedir memoria para el host input data
    h_data = new ubyte[dataLength];

    // Pedir memoria para el host output resultData
    h_resultData = new ubyte[dataLength];

    // Verificar si se inicializaron correctamente 
    if (h_data == NULL || h_resultData == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors:\n");
        exit(EXIT_FAILURE);
    }

    /* initialize random seed: */
    srand(time(NULL));
    // Se inicializan los buffers del host
    initBuffers(h_data, worldWidth, worldHeight);

    // Alojar el device input d_data
    ubyte* d_data = NULL;
    err = cudaMalloc((void**)&d_data, size);

    // Verificar que el vector data se alojo correctamente en el device
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector data (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Alojar el device output d_resultData
    ubyte* d_resultData = NULL;
    err = cudaMalloc((void**)&d_resultData, size);

    // Verificar que el vector resultData se alojo correctamente en el device
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector resultData (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vector data in host memory to the device input vector in device memory
    err = cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector data from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    double time;
    // Launch the game kernel
    time = runGameKernel(d_data, d_resultData, worldWidth, worldHeight, iterations, threads);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch game kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Print the output time
    std::cout << time;

}
