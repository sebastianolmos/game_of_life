"""Grafico con los valores obtenidos en la implementacion serial en CPU"""

from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
import numpy as np

path = "TestProgram/TestProgram/Debug/"

def getValueFromProgram(dim) -> float:
    p = Popen(['TestProgram.exe',  str(dim)], cwd=path, shell=True, stdout=PIPE, stdin=PIPE)
    result = p.stdout.readline().strip()
    return result.decode()

if __name__ == "__main__":
    min = 0
    max = 100000
    interval = 2500

    n = np.arange(min, max + interval, interval)
    t = np.zeros(shape=n.shape)


    for i in range(len(n)):
        t[i] = getValueFromProgram(n[i])

    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot(n, t, 'r-o')

    ax.set(xlabel='Número de celulas', ylabel='tiempo de iteracion (ms)',
       title='Tiempo en que se demora una iteracion para distintos tamaños')
    ax.grid()

    fig.savefig("images/test.png")
    plt.show()