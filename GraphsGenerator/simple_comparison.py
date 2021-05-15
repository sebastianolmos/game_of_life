"""Grafico con los valores obtenidos en la implementacion serial en CPU"""

from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
import numpy as np

path = "TestProgram/TestProgram/Debug/"

def getValueFromProgram1(dim) -> float:
    p = Popen(['TestProgram.exe',  str(dim)], cwd=path, shell=True, stdout=PIPE, stdin=PIPE)
    result = p.stdout.readline().strip()
    return result.decode()

def getValueFromProgram2(dim) -> float:
    p = Popen(['TestProgram.exe',  str(dim)], cwd=path, shell=True, stdout=PIPE, stdin=PIPE)
    result = p.stdout.readline().strip()
    return result.decode()

if __name__ == "__main__":
    min = 0
    max = 100000
    interval = 2500

    n = np.arange(min, max + interval, interval)
    t1 = np.zeros(shape=n.shape)
    t2 = np.zeros(shape=n.shape)


    for i in range(len(n)):
        t1[i] = getValueFromProgram1(n[i])
        t2[i] = getValueFromProgram2(n[i])

    fig, ax = plt.subplots(figsize=(10,7))

    line1 = ax.plot(n, t1, 'r-o', label='Using program 1')
    line2 = ax.plot(n, t2, 'b-o', label='Using program 2')

    ax.set(xlabel='Número de celulas', ylabel='tiempo de iteracion (ms)',
       title='Comparacion de tiempo en que se demora una iteracion para distintos tamaños')
    ax.grid()
    
    ax.legend()
    fig.savefig("images/comparison.png")

    plt.show()