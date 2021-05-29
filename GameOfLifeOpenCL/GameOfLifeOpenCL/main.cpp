#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <stdlib.h>
#include <time.h> 
#include <chrono>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif

typedef unsigned char ubyte;
typedef unsigned short ushort;
typedef unsigned int uint;

void initBuffers(ubyte*& data, size_t worldWidth, size_t worldHeight) {
    size_t dataLength = worldWidth * worldHeight;

    for (size_t i = 0; i < dataLength; i++) {
        data[i] = rand() & 1;
    }
}

int iterator(int argc, char* argv[])
{

	ubyte* h_data;
	ubyte* h_resultData;

	size_t worldHeight;
	size_t worldWidth;
	size_t dataLength;
	size_t iterations;
	ushort threads;

	if (argc < 5)
	{
		worldWidth = 1024;
		worldHeight = 1024;
		iterations = 30;
		threads = 200;
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

	int platform_id = 0, device_id = 0;

	try {
		// Query for platforms
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		// Get a list of devices on this platform
		std::vector<cl::Device> devices;
		// Select the platform.
		platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);

		// Create a context
		cl::Context context(devices);

		// std::cout << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;

		// Create a command queue
		// Select the device.
		cl::CommandQueue queue = cl::CommandQueue(context, devices[device_id]);

		
		// Create the memory buffers on the device
		cl::Buffer d_data = cl::Buffer(context, CL_MEM_READ_WRITE, size);
		cl::Buffer d_resultData = cl::Buffer(context, CL_MEM_READ_WRITE, size);
		cl::Buffer d_worldWidth = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(size_t));
		cl::Buffer d_worldHeight = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(size_t));

		// Copy the input data to the input buffers using the command queue.
		queue.enqueueWriteBuffer(d_data, CL_FALSE, 0, size, h_data);
		queue.enqueueWriteBuffer(d_resultData, CL_FALSE, 0, size, h_resultData);
		queue.enqueueWriteBuffer(d_worldWidth, CL_FALSE, 0, sizeof(size_t), &worldWidth);
		queue.enqueueWriteBuffer(d_worldHeight, CL_FALSE, 0, sizeof(size_t), &worldHeight); // TRUE (?)

		// Read the program source
		std::ifstream sourceFile("kernel.cl");
		std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));

		// Make program from the source code
		cl::Program program = cl::Program(context, source);

		// Build the program for the devices
		program.build(devices);

		// Make kernel
		cl::Kernel gameKernel_kernel(program, "gameKernel");

		// Set the kernel arguments
		// gameKernel_kernel.setArg(0, d_data);
		// gameKernel_kernel.setArg(1, d_resultData);
		// gameKernel_kernel.setArg(2, d_worldWidth);
		// gameKernel_kernel.setArg(3, d_worldHeight);

		double totalTime = 0.0;
		for (size_t i = 0; i < iterations; i++){
			// Set the kernel arguments
			if (i%2){
				gameKernel_kernel.setArg(0, d_data);
				gameKernel_kernel.setArg(1, d_resultData);
			}
			else {
				gameKernel_kernel.setArg(0, d_resultData);
				gameKernel_kernel.setArg(1, d_data);
			}
			gameKernel_kernel.setArg(2, d_worldWidth);
			gameKernel_kernel.setArg(3, d_worldHeight);

			// Execute the kernel
			cl::NDRange global(dataLength);
			cl::NDRange local(threads);

			double itime = 0.0;
			auto start = std::chrono::steady_clock::now();

			queue.enqueueNDRangeKernel(gameKernel_kernel, cl::NullRange, global, local);
			queue.finish();

			auto end = std::chrono::steady_clock::now();
			itime = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
			totalTime += itime;
		}
		/*
		// Execute the kernel
		// cl::NDRange global(dataLength);
		// cl::NDRange local(threads);

		double itime = 0.0;
		auto start = std::chrono::steady_clock::now();

		queue.enqueueNDRangeKernel(gameKernel_kernel, cl::NullRange, global, local);
		queue.finish();

		auto end = std::chrono::steady_clock::now();
		itime = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
		*/
		std::cout << totalTime << std::endl;

		// Copy the output data back to the host
		queue.enqueueReadBuffer(d_resultData, CL_TRUE, 0, size, h_resultData);

		/*
		// Verify the result
		bool result = true;
		for (int i = 0; i < N_ELEMENTS; i++) {
			if (C[i] != A[i] + B[i]) {
				result = false;
				break;
			}
		}
		if (result)
			std::cout << "Success!\n";
		else
			std::cout << "Failed!\n";
		*/
	}
	catch (cl::Error err) {
		std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
		return(EXIT_FAILURE);
	}
}

int main(int argc, char* argv[]) {
	iterator(argc, argv);
	return 0;
}