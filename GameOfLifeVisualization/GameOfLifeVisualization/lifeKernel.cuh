#ifndef LIFE_KERNEL_H_
#define LIFE_KERNEL_H_

#include <stdio.h>
#include <math.h>

#include "helperMath.h"
#include "kernelParams.cuh"


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

	/// CUDA kernel for rendering of life world on the screen.
	/// This kernel transforms bit-per-cell life world to ARGB screen buffer.
__global__ void displayLifeKernel(const ubyte* lifeData, uint worldWidth, uint worldHeight, uchar4* destination,
	int destWidth, int detHeight, bool simulateColors) {

	int multisample = 1;
	bool cyclic = false;
	bool bitLife = false;

	uint pixelId = blockIdx.x * blockDim.x + threadIdx.x;

	int x = (int)floor(((int)(pixelId % destWidth)) * 1.0);
	int y = (int)floor(((int)(pixelId / destWidth)) * 1.0);

	if (cyclic) {
		x = ((x % (int)worldWidth) + worldWidth) % worldWidth;
		y = ((y % (int)worldHeight) + worldHeight) % worldHeight;
	}
	else if (x < 0 || y < 0 || x >= worldWidth || y >= worldHeight) {
		destination[pixelId].x = 127;
		destination[pixelId].y = 127;
		destination[pixelId].z = 127;
		return;
	}

	int value = 0;  // Start at value - 1.
	int increment = 255 / (multisample * multisample);

	if (bitLife) {
		for (int dy = 0; dy < multisample; ++dy) {
			int yAbs = (y + dy) * worldWidth;
			for (int dx = 0; dx < multisample; ++dx) {
				int xBucket = yAbs + x + dx;
				value += ((lifeData[xBucket >> 3] >> (7 - (xBucket & 0x7))) & 0x1) * increment;
			}
		}
	}
	else {
		for (int dy = 0; dy < multisample; ++dy) {
			int yAbs = (y + dy) * worldWidth;
			for (int dx = 0; dx < multisample; ++dx) {
				value += lifeData[yAbs + (x + dx)] * increment;
			}
		}
	}

	bool isNotOnBoundary = !cyclic || !(x == 0 || y == 0);

	if (simulateColors) {
		if (value > 0) {
			if (destination[pixelId].w > 0) {
				// Stayed alive - get darker.
				if (destination[pixelId].y > 145) {
					destination[pixelId].x -= 2;
					destination[pixelId].y -= 2;
					destination[pixelId].z -= 2;
				}
				else
				{
					destination[pixelId].x = 145;
					destination[pixelId].y = 145;
					destination[pixelId].z = 145;
				}
			}
			else {
				// Born - full white color.
				destination[pixelId].x = 255;
				destination[pixelId].y = 255;
				destination[pixelId].z = 255;
			}
		}
		else {
			if (destination[pixelId].w > 0) {
				// Died - dark green.
				destination[pixelId].x = 255;
				destination[pixelId].y = 0;
				destination[pixelId].z = 0;
			}
			else {
				// Stayed dead - get darker.
				if (destination[pixelId].x > 33) {
					if (isNotOnBoundary) {
					}
					destination[pixelId].x -= 16;
				}
				else {
					destination[pixelId].x = 32;
					destination[pixelId].y = 0;
					destination[pixelId].z = 0;
				}
			}
		}
	}
	else {
		destination[pixelId].x = value;
		destination[pixelId].y = value;
		destination[pixelId].z = value;
	}

	// Save last state of the cell to the alpha channel that is not used in rendering.
	destination[pixelId].w = value;
}

/// CUDA kernel for clean a device buffer
__global__ void cleanBufferKernel(uchar4* buffer) {

	uint pixelId = blockIdx.x * blockDim.x + threadIdx.x;

	buffer[pixelId].x = 0;
	buffer[pixelId].y = 0;
	buffer[pixelId].z = 0;
	buffer[pixelId].w = 0;

}


#endif