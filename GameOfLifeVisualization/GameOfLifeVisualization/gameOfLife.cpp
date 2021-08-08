#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shaders/shader.h"
#include "utils/performanceMonitor.h"
#include "utils/controller.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include "gameOfLife.cuh"

#include <iostream>
#include <time.h> 

#define GL_PIXEL_UNPACK_BUFFER_ARB        0x88EC

using namespace std;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void key_callback_wrapper(GLFWwindow* window, int key, int scancode, int action, int mods);
void initBuffers(ubyte*& data, size_t worldWidth, size_t worldHeight, int dist);

// settings
unsigned int SCR_WIDTH = 800;
unsigned int SCR_HEIGHT = 800;
float IterPerSeconds = 20.0f;
float deltaTime = 0.0f;
glm::vec3 translate = glm::vec3(0.0f);
float scale = 1.0f;
bool simulateColors = true;

Controller* controller = new Controller(1.0f, 7.0f, 10.0f);


int main(int argc, char* argv[])
{
    // Error code to check return values for CUDA calls
    //cudaError_t err = cudaSuccess;

    ubyte* h_data;
    ubyte* h_resultData;

    size_t worldHeight;
    size_t worldWidth;
    size_t dataLength;
    size_t iterations;
    ushort threads;
    uint distance;

    if (argc < 5)
    {
        worldWidth = 128;
        worldHeight = 128;
        threads = 64;
        distance = 32;
    }
    else {
        worldWidth = atoi(argv[1]);
        worldHeight = atoi(argv[2]);
        threads = atoi(argv[4]);
        distance = atoi(argv[5]);
    }
    SCR_WIDTH = (int)(worldWidth/ worldHeight)*SCR_WIDTH;

    // Paramas
    // -----------------------------
    /// Host-side texture pointer.
    uchar4* h_textureBufferData = nullptr;
    /// Device-side texture pointer.
    uchar4* d_textureBufferData = nullptr;

    GLuint gl_pixelBufferObject = 0;
    GLuint gl_texturePtr = 0;
    cudaGraphicsResource* cudaPboResource = nullptr;

    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


    string title = "Game of Life";
    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, title.c_str(), NULL, NULL);
    if (window == NULL)
    {
        cout << "Failed to create GLFW window" << endl;
        glfwTerminate();
        return -1;
    }

    // Controller Init
    // -------
    controller->setScale(&scale);
    controller->setTranslate(&translate);
    controller->setIterPerSecond(&IterPerSeconds);

    // GLFW callbacks
    // -------
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_callback_wrapper);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        cout << "Failed to initialize GLAD" << endl;
        return -1;
    }


    // Initialiaze host and device buffers
    // ------------------------------

    dataLength = worldWidth * worldHeight;
    size_t size = dataLength * sizeof(ubyte);

    // Pedir memoria para el host input data
    h_data = new ubyte[dataLength];

    // Pedir memoria para el host output resultData
    h_resultData = new ubyte[dataLength];

    /* initialize random seed: */
    srand(time(NULL));
    // Se inicializan los buffers del host
    initBuffers(h_data, worldWidth, worldHeight, distance);

    // Alojar el device input d_data
    ubyte* d_data = NULL;
    allocateArray((void**)&d_data, size);

    // Alojar el device output d_resultData
    ubyte* d_resultData = NULL;
    allocateArray((void**)&d_resultData, size);

    copyArrayToDevice(d_data, h_data, 0, size);

    // build and compile our shader program
    // ------------------------------------
    Shader textureShader("shaders/transformTexShader.vs", "shaders/transformTexShader.fs"); // you can name your shader files however you like

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float rad = 1.0f;
    float vertices[] = {
        // positions        // texture coords
         rad,  rad, 0.0f, 1.0f, 0.0f, // top right
         rad, -rad, 0.0f, 1.0f, 1.0f, // bottom right
        -rad, -rad, 0.0f, 0.0f, 1.0f, // bottom left
        -rad,  rad, 0.0f, 0.0f, 0.0f  // top left 
    };

    unsigned int indices[] = {  // note that we start from 0!
        0, 1, 3,  // first Triangle
        1, 2, 3   // second Triangle
    };

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
    // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
    // glBindVertexArray(0);

    delete[] h_textureBufferData;
    h_textureBufferData = nullptr;

    glDeleteTextures(1, &gl_texturePtr);
    gl_texturePtr = 0;

    if (gl_pixelBufferObject) {
        cudaGraphicsUnregisterResource(cudaPboResource);
        glDeleteBuffers(1, &gl_pixelBufferObject);
        gl_pixelBufferObject = 0;
    }

    // CUDA Interp intialization
    // -------------------------
    h_textureBufferData = new uchar4[worldWidth * worldHeight];

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_texturePtr);
    glBindTexture(GL_TEXTURE_2D, gl_texturePtr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, worldWidth, worldHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_textureBufferData);

    glGenBuffers(1, &gl_pixelBufferObject);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, worldWidth* worldHeight * sizeof(uchar4), h_textureBufferData, GL_STREAM_COPY);

    // While a PBO is registered to CUDA, it can't be used as the destination for OpenGL drawing calls.
    // But in our particular case OpenGL is only used to display the content of the PBO, specified by CUDA kernels,
    // so we need to register/unregister it only once.
    cudaError result = cudaGraphicsGLRegisterBuffer(&cudaPboResource, gl_pixelBufferObject, cudaGraphicsMapFlagsWriteDiscard);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    float t1 = (float)glfwGetTime();
    float t0 = (float)glfwGetTime();
    float timeCounter = 0.0f;
    bool swapBuffers = false;

    // CLEAN the device Buffer before render
    // -----------
    cudaGraphicsMapResources(1, &cudaPboResource, 0); // PBO Map
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_textureBufferData, &num_bytes, cudaPboResource);
    cleanBufferInDevice(d_textureBufferData, worldWidth, worldHeight); // clean Buffer kernel call
    cudaGraphicsUnmapResources(1, &cudaPboResource, 0); // PBO UnMap

    PerformanceMonitor pMonitor(glfwGetTime(), 0.5f);


    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        float time = (float)glfwGetTime();
        t1 = time;
        deltaTime = t1 - t0;
        t0 = t1;
        timeCounter += deltaTime;

        // Performance Monitor Update
        // -----------------
        pMonitor.update(time, ceil(IterPerSeconds));
        stringstream ss;
        ss << title << " " << pMonitor;
        glfwSetWindowTitle(window, ss.str().c_str());

        // Controller Update
        // -----------------
        controller->updateParams(deltaTime);

        // Execute the game Kernel according to the iterations per frame value
        // -----------------
        if (timeCounter > 1.0 / IterPerSeconds) {
            //Game kernel call
            // -----
            runGameInDevice(d_data, d_resultData, worldWidth, worldHeight, threads);
            swapBuffers = true;

            // PBO Map
            // ------
            cudaGraphicsMapResources(1, &cudaPboResource, 0);
            size_t num_bytes;
            cudaGraphicsResourceGetMappedPointer((void**)&d_textureBufferData, &num_bytes, cudaPboResource);

            // Display kernel call
            // ------
            runDisplayLifeKernel(d_resultData, worldWidth, worldHeight, d_textureBufferData, worldWidth, worldHeight, simulateColors);

            // PBO UnMap
            // ------
            cudaGraphicsUnmapResources(1, &cudaPboResource, 0);

            timeCounter = 0;
        }
        

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Bind the texture and PBO
        // ---------
        glBindTexture(GL_TEXTURE_2D, gl_texturePtr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);

        // copy pixels from PBO to texture object
        // Use offset instead of ponter.
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, worldWidth, worldHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);

        // transformation 
        // --------
        glm::mat4 transform = glm::mat4(1.0f);
        transform = glm::translate(transform, translate);
        transform = glm::scale(transform, glm::vec3(scale, scale, 1.0f));

        // render the quad
        // --------
        textureShader.use();
        textureShader.setMat4("transform", transform);

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // Unbind the texture and PBO
        // ---------
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        glBindTexture(GL_TEXTURE_2D, 0);


        // Buffer swap
        // -----
        if (swapBuffers)
        {
            swap(d_data, d_resultData);
            swapBuffers = false;
        }

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    
    // FREE Buffers
    // --------------------------

    delete[] h_data;
    h_data = nullptr;
    delete[] h_resultData;
    h_resultData = nullptr;

    cudaFree(d_data);
    d_data = nullptr;
    cudaFree(d_resultData);
    d_resultData = nullptr;

    delete[] h_textureBufferData;
    h_textureBufferData = 0;
    cudaFree(d_textureBufferData);
    d_textureBufferData = nullptr;

    glDeleteTextures(1, &gl_texturePtr);
    glDeleteBuffers(1, &gl_pixelBufferObject);
    cudaPboResource = nullptr;

    return 0;
}

// Wrapper to call controller method
// ----------------------------------------
void key_callback_wrapper(GLFWwindow* window, int key, int scancode, int action, int mods) {
    controller->key_callback(window, key, scancode, action, mods);
    if (action == GLFW_PRESS && key == GLFW_KEY_P)
    {
        simulateColors = !simulateColors;
    }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

void initBuffers(ubyte*& data, size_t worldWidth, size_t worldHeight, int dist) {
    size_t dataLength = worldWidth * worldHeight;

    for (size_t i = 0; i < dataLength; i++) {
        int x = (int)floor(((int)(i % worldWidth)) * 1.0);
        int y = (int)floor(((int)(i / worldHeight)) * 1.0);
        int centerX = (int)(worldWidth / 2.0f);
        int centerY = (int)(worldHeight / 2.0f);
        if (((centerX - x > -dist) && (centerX - x < dist)) &&
            ((centerY - y > -dist) && (centerY - y < dist)))
        {
            data[i] = rand() & 1;
        }
        else {
            data[i] = 0;
        }
    }
}