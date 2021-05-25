"""Grafico con los valores obtenidos en la implementacion serial en CPU2 con ifs"""

import matplotlib.pyplot as plt
import numpy as np
import csv

path = "Data/"

if __name__ == "__main__":
    size = []
    size2 = []
    time1 = []
    time2 = []
    time3 = []

    with open(path + 'serial_CPU.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            size += [ int(row['width']) * int(row['height']) ]
            tmp_time = float(row['time']) * 0.001
            time1 += [tmp_time / float(row['iter']) ]
            print(size[-1], time1[-1])

    with open(path + 'cuda_128tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            size2 += [ int(row['width']) * int(row['height']) ]
            tmp_time = float(row['time']) * 0.001
            time2 += [tmp_time / float(row['iter']) ]
            print(size2[-1], time2[-1])

    with open(path + 'opencl_256tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            tmp_time = float(row['time']) * 0.001
            time3 += [tmp_time / float(row['iter']) ]
            print(time2[-1])

    n_size = np.array(size)
    n_size2 = np.array(size2)
    n_time1 = np.array(time1) 
    n_time2 = np.array(time2) 
    n_time3 = np.array(time3) 

    n_eval1 = (n_size / n_time1) / 1000000
    n_eval2 = (n_size2 / n_time2) / 1000000
    n_eval3 = (n_size2 / n_time3) / 1000000

    fig, ax = plt.subplots(figsize=(10,7))
    
    ax.set_xscale('log')

    line1 = ax.plot(n_size, n_eval1, 'r-o', label='CPU')
    line2 = ax.plot(n_size2, n_eval2, 'b-o', label='CUDA')
    line3 = ax.plot(n_size2, n_eval3, 'g-o', label='OpenCL')

    ax.set(xlabel='Tamaño del mundo [Células]', ylabel='Células evaluadas por segundo [millones]',
       title='Comparación de tiempo promedio de iteración entre implementaciones serial en CPU\n y paralelas en CUDA y OpenCL')
    ax.grid()

    ax.legend()
    fig.savefig("images/CPU_CUDA_OpenCL_comparison.png")
    plt.show()