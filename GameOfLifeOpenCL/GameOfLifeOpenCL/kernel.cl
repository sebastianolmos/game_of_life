kernel void gameKernel(global const unsigned char* data, global unsigned char* resultData, global const unsigned int* worldWidth, global const unsigned int* worldHeight)
{   
    unsigned int worldWidthValue = *worldWidth;
    unsigned int worldHeightValue = *worldHeight;
    unsigned int worldSize = worldWidthValue * worldHeightValue;
    
    unsigned int cell = get_group_id(0) * get_local_size(0) + get_local_id(0);
    for (cell; cell < worldSize; cell += get_local_size(0)  * get_num_groups(0)) {
        unsigned int x = cell % worldWidthValue;
        unsigned int y = cell - x;
        unsigned int xLeft = (x + worldWidthValue - 1) % worldWidthValue;
        unsigned int xRight = (x + 1) % worldWidthValue;
        unsigned int yUp = (y + worldSize - worldWidthValue) % worldSize;
        unsigned int yDown = (y + worldWidthValue) % worldSize;

        unsigned int aliveCells = data[xLeft + yUp] + data[x + yUp] + data[xRight + yUp] + data[xLeft + y]
            + data[xRight + y] + data[xLeft + yDown] + data[x + yDown] + data[xRight + yDown];

        resultData[x + y] = aliveCells == 3 || (aliveCells == 2 && data[x + y]) ? 1 : 0;
    }
}