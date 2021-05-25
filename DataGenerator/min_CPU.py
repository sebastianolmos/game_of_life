"""Generar data de la implementacion serial en CPU"""

from subprocess import Popen, PIPE
import numpy as np
import csv

path = "GameOfLifeCPU/GameOfLifeCPU/x64/Release/"

def getValueFromProgram(width, height, iterations) -> float:
    p = Popen(['GameOfLifeCPU.exe',  str(width), str(height), str(iterations)], cwd=path, shell=True, stdout=PIPE, stdin=PIPE)
    result = p.stdout.readline().strip()
    return result.decode()

if __name__ == "__main__":

    width = 2
    height = 2
    worlds = 22
    iterations = 5

    with open('Data/serial_min_CPU.csv', mode='w') as csv_file:
        fieldnames = ['width', 'height', 'iter', 'time']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()

        for i in range(worlds):
            if (i % 2):
                height *= 2
            else:
                width *= 2
            time = getValueFromProgram(width, height, iterations) 
            writer.writerow({'width': width, 'height': height, 'iter': iterations, 'time': float(time)*1e-6})
