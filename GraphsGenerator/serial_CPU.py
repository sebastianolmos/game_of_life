"""Grafico con los valores obtenidos en la implementacion serial en CPU"""

import matplotlib.pyplot as plt
import numpy as np
import csv

path = "Data/"

if __name__ == "__main__":
    size = []
    time = []

    with open(path + 'serial_CPU.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            size += [ int(row['width']) * int(row['height']) ]
            tmp_time = float(row['time']) * 0.001
            time += [tmp_time / float(row['iter']) ]
            print(size[-1], time[-1])

    n_size = np.array(size)
    n_time = np.array(time) 
    print(n_size)
    print(n_time)

    n_eval = n_size / n_time / 1000000

    print(n_eval)

    fig, ax = plt.subplots(figsize=(10,7))
    ax.set_xscale('log')
    ax.plot(n_size, n_eval, 'r-o')

    ax.set(xlabel='Tamaño del mundo [Células]', ylabel='Células evaluadas por segundo [Millones]',
       title='Células evaluadas por segundo para distintos tamaños de mundo\ncon implementación secuencial en CPU ')
    ax.grid()

    fig.savefig("images/serial_CPU.png")
    plt.show()