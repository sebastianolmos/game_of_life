"""Generar data de la implementacion serial en CPU 2 con ifs statements"""

from subprocess import Popen, PIPE
import numpy as np
import csv

path = "GameOfLifeCPU2/GameOfLifeCPU2/x64/Release/"

def getValueFromProgram(width, height, iterations) -> float:
    p = Popen(['GameOfLifeCPU2.exe',  str(width), str(height), str(iterations)], cwd=path, shell=True, stdout=PIPE, stdin=PIPE)
    result = p.stdout.readline().strip()
    return result.decode()

if __name__ == "__main__":

    width = 64
    height = 64
    worlds = 20
    iterations = 5



    with open('Data/serial_CPU2.csv', mode='w') as csv_file:
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
