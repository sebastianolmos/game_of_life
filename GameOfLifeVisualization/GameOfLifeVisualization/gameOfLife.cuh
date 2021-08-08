#include "kernelParams.cuh"

extern "C"
{
    void allocateArray(void** devPtr, size_t size);

    void freeArray(void* devPtr);

    void threadSync();

    void copyArrayToDevice(void* device, const void* host, int offset, int size);

    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource** cuda_vbo_resource);

    void unregisterGLBufferObject(struct cudaGraphicsResource* cuda_vbo_resource);

    void* mapGLBufferObject(struct cudaGraphicsResource** cuda_vbo_resource);

    void unmapGLBufferObject(struct cudaGraphicsResource* cuda_vbo_resource);

    void copyArrayFromDevice(void* host, const void* device, struct cudaGraphicsResource** cuda_vbo_resource, int size);

    void runGameInDevice(ubyte*& d_data, ubyte*& d_resultData, size_t worldWidth, size_t worldHeight, ushort threads);

    void runDisplayLifeKernel(const ubyte* d_lifeData, size_t worldWidth, size_t worldHeight, uchar4* destination,
        int destWidth, int destHeight, bool simulateColors);

    void cleanBufferInDevice(uchar4* buffer, size_t width, size_t height);
}
