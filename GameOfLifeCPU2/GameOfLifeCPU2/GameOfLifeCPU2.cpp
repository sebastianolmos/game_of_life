// GameOfLifeCPU2.cpp : This file contains the 'main' function. Program execution begins and ends there.
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
size_t iterations;


void initBuffers() {
    data = new ubyte[worldHeight * worldWidth];
    resultData = new ubyte[worldHeight * worldWidth];
    dataLength = worldWidth * worldHeight;

    for (size_t i = 0; i < dataLength; i++) {
        data[i] = rand() & 1;
        resultData[i] = 0;
    }
}

ubyte countAliveCells(size_t x0, size_t x1, size_t x2, size_t y0, size_t y1, size_t y2) {
    ubyte result = 0;
    if (data[x0 + y0] != 0) {
        result = result + 1;
    }
    if (data[x1 + y0] != 0) {
        result = result + 1;
    }
    if (data[x2 + y0] != 0) {
        result = result + 1;
    }
    if (data[x0 + y1] != 0) {
        result = result + 1;
    }
    if (data[x2 + y1] != 0) {
        result = result + 1;
    }
    if (data[x0 + y2] != 0) {
        result = result + 1;
    }
    if (data[x1 + y2] != 0) {
        result = result + 1;
    }
    if (data[x2 + y2] != 0) {
        result = result + 1;
    }
    return result;
}

double worldIteration() {
    double itime = 0.0;
    auto start = std::chrono::steady_clock::now();

    for (size_t j = 0; j < worldHeight; j++) {
        size_t y0 = ((j - 1 + worldHeight) % worldHeight) * worldWidth;
        size_t y1 = j * worldWidth;
        size_t y2 = ((j + 1) % worldHeight) * worldWidth;
        for (size_t i = 0; i < worldWidth; i++) {
            size_t x = i;
            size_t x0 = (x - 1 + worldWidth) % worldHeight;
            size_t x2 = (x + 1) % worldWidth;

            ubyte aliveCells = countAliveCells(x0, x, x2, y0, y1, y2);
            if (aliveCells == 3 || (aliveCells == 2 && data[y1 + x])) {
                resultData[y1 + x] = 1;
            }
            else {
                resultData[y1 + x] = 0;
            }
        }
    }
    auto end = std::chrono::steady_clock::now();
    itime = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::swap(data, resultData);
    return itime;
}

double runGameLife() {
    double totalTime = 0.0;
    for (size_t i = 0; i < iterations; i++) {
        totalTime += worldIteration();
    }
    return totalTime;
}

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        worldWidth = 1000;
        worldHeight = 1000;
        iterations = 1;
    }
    else {
        worldWidth = atoi(argv[1]);
        worldHeight = atoi(argv[2]);
        iterations = atoi(argv[3]);
    }
    /* initialize random seed: */
    srand(time(NULL));

    // Se inicializan los buffers
    initBuffers();

    // Se ejecuta el juego
    double gameTime = runGameLife();
    // nanoseconds
    // microseconds
    // milliseconds

    std::cout << gameTime;
}