// GameOfLifeCPU.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <stdio.h>     
#include <stdlib.h>
#include <iostream>
#include <time.h> 
#include <chrono>

typedef unsigned char ubyte;
ubyte* data = NULL;
ubyte* resultData;

size_t worldHeight;
size_t worldWidth;
size_t dataLength;


void initBuffers() {
    data = new ubyte[worldHeight * worldWidth];
    resultData = new ubyte[worldHeight * worldWidth];
    dataLength = worldWidth * worldHeight;

    for (size_t i = 0; i < worldHeight; i++) {
        data[i] = rand() & 1;
        resultData[i] = 0;
    }
}

ubyte countAliveCells(size_t x0, size_t x1, size_t x2, size_t y0, size_t y1, size_t y2) {
    ubyte result = data[x0 + y0] + data[x1 + y0] + data[x2 + y0] + data[x0 + y1]
        + data[x2 + y1] + data[x0 + y2] + data[x1 + y2] + data[x2 + y2];
    return result;
}

void worldIteration() {
    for (size_t j = 0; j < worldHeight; j++) {
        size_t y0 = ((j - 1 + worldHeight) % worldHeight) * worldWidth;
        size_t y1 = j * worldWidth;
        size_t y2 = ((j + 1) % worldHeight) * worldWidth;
        for (size_t i = 0; i < worldWidth; i++) {
            size_t x = i;
            size_t x0 = (x - 1 + worldWidth) % worldHeight;
            size_t x2 = (x + 1) % worldWidth;

            ubyte aliveCells = countAliveCells(x0, x, x2, y0, y1, y2);
            resultData[y1 + x] = aliveCells == 3 || (aliveCells == 2 && data[y1 + x]) ? 1 : 0;
        }
    }
    std::swap(data, resultData);
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        worldWidth = 100;
        worldHeight = 100;
    }
    else {
        worldWidth = atoi(argv[1]);
        worldHeight = atoi(argv[2]);
    }
    /* initialize random seed: */
    srand(time(NULL));

    // Se inicializan los buffers
    initBuffers();

    auto start = std::chrono::steady_clock::now();
    // Se ejecuta una iteracion
    worldIteration();
    auto end = std::chrono::steady_clock::now();
    // nanoseconds
    // microseconds
    // milliseconds

    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}
