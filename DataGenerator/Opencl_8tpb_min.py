"""Generar data de la implementacion en GPU"""

from subprocess import Popen, PIPE
import numpy as np
import csv

path = "GameOfLifeCUDA/GameOfLifeCUDA/x64/Release/"
path = "GameOfLifeOpenCL/GameOfLifeOpenCL/x64/Release/"

def getValueFromProgram(width, height, iterations, threads) -> float:
    p = Popen(['GameOfLifeOpenCL.exe',  str(width), str(height), str(iterations), str(threads)], cwd=path, shell=True, stdout=PIPE, stdin=PIPE)
    result = p.stdout.readline().strip()
    return result.decode()

if __name__ == "__main__":

    width = 2
    height = 2
    worlds = 22
    iterations = 5
    threads = 8

    with open('Data/opencl_8tpb_min.csv', mode='w') as csv_file:
        fieldnames = ['width', 'height', 'iter', 'time']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()

        for i in range(worlds):
            if (i % 2):
                if(i < worlds - 1): 
                    height *= 2
                else:
                    break
            else:
                width *= 2
            time = getValueFromProgram(width, height, iterations, threads) 
            writer.writerow({'width': width, 'height': height, 'iter': iterations, 'time': float(time)*1e-6})
