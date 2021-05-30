"""Grafico con los valores obtenidos en las implementacion serial y paralelas"""

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
    time4 = []
    time5 = []

    with open(path + 'serial_CPU.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            size += [ int(row['width']) * int(row['height']) ]
            tmp_time = float(row['time']) * 0.001
            time1 += [tmp_time / float(row['iter']) ]
            print(size[-1], time1[-1])
    
    with open(path + 'non_32_CPU.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            size2 += [ int(row['width']) * int(row['height']) ]
            tmp_time = float(row['time']) * 0.001
            time2 += [tmp_time / float(row['iter']) ]
            print(size2[-1], time2[-1])

    with open(path + 'cuda_128tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            tmp_time = float(row['time']) * 0.001
            time3 += [tmp_time / float(row['iter']) ]
            print(time3[-1])

    with open(path + 'cuda_128tpb_if.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            tmp_time = float(row['time']) * 0.001
            time4 += [tmp_time / float(row['iter']) ]
            print(time4[-1])
    
    with open(path + 'cuda_81tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            tmp_time = float(row['time']) * 0.001
            time5 += [tmp_time / float(row['iter']) ]
            print(time5[-1])

    n_size = np.array(size)[:-1]
    n_size2 = np.array(size2)
    n_time1 = np.array(time1)[:-1]
    n_time2 = np.array(time2) 
    n_time3 = np.array(time3) 
    n_time4 = np.array(time4) 
    n_time5 = np.array(time5) 

    n_eval1 = (n_time1 / n_time3)
    n_eval2 = (n_time1 / n_time4)
    n_eval3 = (n_time2 / n_time5)

    fig, ax = plt.subplots(figsize=(10,7))
    
    ax.set_xscale('log')

    line1 = ax.plot(n_size, n_eval1, 'r-o', label='CUDA')
    line2 = ax.plot(n_size, n_eval2, 'b-o', label='CUDA con ifs')
    line3 = ax.plot(n_size2, n_eval3, 'g-o', label='CUDA no múltiplo de 32')

    ax.set(xlabel='Tamaño del mundo [Células]', ylabel='Speedup [n-veces]',
       title='Speedup para implementación en CUDA y sus variaciones sobre CPU')
    ax.grid()

    ax.legend()
    fig.savefig("images/CPU_CUDA_variations_speedup.png")
    plt.show()