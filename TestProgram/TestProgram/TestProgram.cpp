// TestProgram.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <stdlib.h>
#include <windows.h>
#include <ctime>
#include <random>


float getValue(int max)
{   
    std::mt19937_64 gen{ std::random_device()() };
    std::uniform_real_distribution<double> dis{ 0.0, double(max) };
    float value = float (dis(gen));
    return value;
}

int main(int argc, char* argv[])
{
    int input;
    if (argc < 2) 
    {
        input = 1;
    }
    else {
        input = atoi(argv[1]);
    }
    float output = getValue(input);
    std::cout << output << std::endl;
}