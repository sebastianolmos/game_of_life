#include <glad/glad.h>

#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>

#include <stdio.h>
#include <assert.h>

#include "lifeKernel.cuh"


extern "C"
{
    void allocateArray(void** devPtr, size_t size)
    {
        cudaMalloc(devPtr, size);
    }

    void freeArray(void* devPtr)
    {
        cudaFree(devPtr);
    }

    void threadSync()
    {
        cudaDeviceSynchronize();
    }

    void copyArrayToDevice(void* device, const void* host, int offset, int size)
    {
        cudaMemcpy((char*)device + offset, host, size, cudaMemcpyHostToDevice);
    }

    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource** cuda_vbo_resource)
    {
        cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);
    }

    void unregisterGLBufferObject(struct cudaGraphicsResource* cuda_vbo_resource)
    {
        cudaGraphicsUnregisterResource(cuda_vbo_resource);
    }

    void* mapGLBufferObject(struct cudaGraphicsResource** cuda_vbo_resource)
    {
        void* ptr;
        cudaGraphicsMapResources(1, cuda_vbo_resource, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&ptr, &num_bytes, *cuda_vbo_resource);
        return ptr;
    }

    void unmapGLBufferObject(struct cudaGraphicsResource* cuda_vbo_resource)
    {
        cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
    }

    void copyArrayFromDevice(void* host, const void* device,
        struct cudaGraphicsResource** cuda_vbo_resource, int size)
    {
        if (cuda_vbo_resource)
        {
            device = mapGLBufferObject(cuda_vbo_resource);
        }

        cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);

        if (cuda_vbo_resource)
        {
            unmapGLBufferObject(*cuda_vbo_resource);
        }
    }

    //Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint& numBlocks, uint& numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    void runGameInDevice(ubyte*& d_data, ubyte*& d_resultData, size_t worldWidth, size_t worldHeight, ushort threads)
    {
        // threads
        size_t calcBlocks = (worldWidth * worldHeight) / threads;
        ushort blocks = (ushort)min((size_t)32768, calcBlocks);

        // set all cells to empty
        //cudaMemset(d_resultData, 0x00000000, worldWidth * worldHeight * sizeof(ubyte));
        // execute the kernel
        gameKernel << <blocks, threads >> > ((ubyte*)d_data, (ubyte*)d_resultData, worldWidth, worldHeight);
    }

    /// Runs a kernel for rendering of life world on the screen.
    void runDisplayLifeKernel(const ubyte* d_lifeData, size_t worldWidth, size_t worldHeight, uchar4* destination,
        int destWidth, int destHeight, bool simulateColors) {

        ushort threadsCount = 256;
        assert((worldWidth * worldHeight) % threadsCount == 0);
        size_t reqBlocksCount = (destWidth * destHeight) / threadsCount;
        assert(reqBlocksCount < 65536);
        ushort blocksCount = (ushort)reqBlocksCount;
        displayLifeKernel << <blocksCount, threadsCount >> > ((ubyte*)d_lifeData, uint(worldWidth), uint(worldHeight), (uchar4*)destination,
            destWidth, destHeight, simulateColors);
        cudaDeviceSynchronize();
    }

    /// Runs a kernel for cleaning 
    void cleanBufferInDevice(uchar4* buffer, size_t width, size_t height) {

        ushort threadsCount = 256;
        assert((worldWidth * worldHeight) % threadsCount == 0);
        size_t reqBlocksCount = (width * height) / threadsCount;
        assert(reqBlocksCount < 65536);
        ushort blocksCount = (ushort)reqBlocksCount;
        cleanBufferKernel << <blocksCount, threadsCount >> >(  (uchar4*)buffer);
        cudaDeviceSynchronize();
    }

}
