"""Grafico con los valores obtenidos en la implementacion serial en CPU2 con ifs"""

import matplotlib.pyplot as plt
import numpy as np
import csv

path = "Data/"

if __name__ == "__main__":
    size = []
    time1 = []
    time2 = []

    with open(path + 'serial_CPU.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            size += [ int(row['width']) * int(row['height']) ]
            tmp_time = float(row['time']) * 0.001
            time1 += [tmp_time / float(row['iter']) ]
            print(size[-1], time1[-1])

    with open(path + 'serial_CPU2.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            # size = int(row['width']) * int(row['height']) 
            tmp_time = float(row['time']) * 0.001
            time2 += [tmp_time / float(row['iter']) ]
            print(time2[-1])

    n_size = np.array(size)
    n_time1 = np.array(time1) 
    n_time2 = np.array(time2) 

    n_eval1 = n_size / n_time1
    n_eval2 = n_size / n_time2

    fig, ax = plt.subplots(figsize=(10,7))
    
    ax.set_xscale('log')

    line1 = ax.plot(n_size, n_eval1, 'r-o', label='CPU implementation')
    line2 = ax.plot(n_size, n_eval2, 'b-o', label='CPU ifs implementation')

    ax.set(xlabel='Número de celulas', ylabel='Celulas evaluadas por segundo (tpb)',
       title='Comparacion de tiempo promedio de iteracion entre implementaciones en CPU')
    ax.grid()

    ax.legend()
    fig.savefig("images/CPU_comparison.png")
    plt.show()