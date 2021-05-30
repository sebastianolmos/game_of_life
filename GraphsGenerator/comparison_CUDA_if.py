"""Grafico con la comparacion de diferentes tpb usando la implementacion con CUDA"""

import matplotlib.pyplot as plt
import numpy as np
import csv

path = "Data/"

if __name__ == "__main__":
    size = []
    size_if = []
    time = []
    time_if = []

    with open(path + 'cuda_128tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            size += [ int(row['width']) * int(row['height']) ]
            tmp_time = float(row['time']) * 0.001
            time += [tmp_time / float(row['iter']) ]
            print(size[-1], time[-1])

    with open(path + 'cuda_128tpb_if.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            size_if += [ int(row['width']) * int(row['height']) ]
            tmp_time = float(row['time']) * 0.001
            time_if += [tmp_time / float(row['iter']) ]
            print(size_if[-1], time_if[-1])

    n_size = np.array(size)
    n_time = np.array(time)
    n_time_if = np.array(time_if)

    n_eval = n_size / n_time / 1000000
    n_eval_if = n_size / n_time_if / 1000000

    fig, ax = plt.subplots(figsize=(10,7))
    
    ax.set_xscale('log')

    line = ax.plot(n_size, n_eval, 'b-o', label='Implementación sin if statements')
    line_if = ax.plot(n_size, n_eval_if, 'r-o', label=' Implementación con if statements')

    ax.set(xlabel='Tamaño del mundo [Células]', ylabel='Células evaluadas por segundo [Millones]',
       title='Comparación de células evaluadas por segundo entre implementaciones\nparalelas en CUDA y su variación con if statements')
    ax.grid()

    ax.legend()
    fig.savefig("images/CUDA_if_comparison.png")
    plt.show()