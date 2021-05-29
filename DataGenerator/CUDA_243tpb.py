"""Generar data de la implementacion serial en CPU"""

from subprocess import Popen, PIPE
import numpy as np
import csv

path = "GameOfLifeCUDA/GameOfLifeCUDA/x64/Release/"

def getValueFromProgram(width, height, iterations, threads) -> float:
    p = Popen(['GameOfLifeCUDA.exe',  str(width), str(height), str(iterations), str(threads)], cwd=path, shell=True, stdout=PIPE, stdin=PIPE)
    result = p.stdout.readline().strip()
    return result.decode()

if __name__ == "__main__":

    width = 27
    height = 27
    worlds = 14
    iterations = 5
    threads = 243

    with open('Data/cuda_243tpb.csv', mode='w') as csv_file:
        fieldnames = ['width', 'height', 'iter', 'time']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()

        for i in range(worlds):
            if (i % 2):
                if(i < worlds - 1): 
                    height *= 3
                else:
                    break
            else:
                width *= 3
            
            time = getValueFromProgram(width, height, iterations, threads)
            writer.writerow({'width': width, 'height': height, 'iter': iterations, 'time': float(time)})
