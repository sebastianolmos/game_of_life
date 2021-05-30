"""Grafico con los valores obtenidos en la implementacion serial en CPU y OpenCL"""

import matplotlib.pyplot as plt
import numpy as np
import csv

path = "Data/"

if __name__ == "__main__":
    size1 = []
    size2 = []
    size3 = []
    size4 = []
    time1 = []
    time2 = []
    time3 = []
    time4 = []

    with open(path + 'serial_min_CPU.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            size1 += [ int(row['width']) * int(row['height']) ]
            tmp_time = float(row['time']) * 0.001
            time1 += [tmp_time / float(row['iter']) ]
            print(size1[-1], time1[-1])

    with open(path + 'opencl_8tpb_min.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            size2 += [ int(row['width']) * int(row['height']) ]
            tmp_time = float(row['time']) * 0.001
            time2 += [tmp_time / float(row['iter']) ]
            print(size2[-1], time2[-1])

    with open(path + 'opencl_16tpb_min.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            size3 += [ int(row['width']) * int(row['height']) ]
            tmp_time = float(row['time']) * 0.001
            time3 += [tmp_time / float(row['iter']) ]
            print(size3[-1], time2[-1])
    
    with open(path + 'opencl_32tpb_min.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            size4 += [ int(row['width']) * int(row['height']) ]
            tmp_time = float(row['time']) * 0.001
            time4 += [tmp_time / float(row['iter']) ]
            print(size4[-1], time2[-1])

    n_size1 = np.array(size1)
    n_size2 = np.array(size2)
    n_size3 = np.array(size3)
    n_size4 = np.array(size4)
    n_time1 = np.array(time1) 
    n_time2 = np.array(time2) 
    n_time3 = np.array(time3)
    n_time4 = np.array(time4)

    n_eval1 = (n_size1 / n_time1) / 1000000
    n_eval2 = (n_size2 / n_time2) / 1000000
    n_eval3 = (n_size3 / n_time3) / 1000000
    n_eval4 = (n_size4 / n_time4) / 1000000

    fig, ax = plt.subplots(figsize=(10,7))
    
    ax.set_xscale('log')

    line1 = ax.plot(n_size1, n_eval1, 'r-o', label='CPU')
    line2 = ax.plot(n_size2, n_eval2, 'b-o', label='8 tpb')
    line3 = ax.plot(n_size3, n_eval3, 'g-o', label='16 tpb')
    line3 = ax.plot(n_size4, n_eval4, 'y-o', label='32 tpb')

    ax.set(xlabel='Tamaño del mundo [Células]', ylabel='Células evaluadas por segundo [Millones]',
       title='Comparación de células evaluadas por segundo entre implementaciones serial en CPU\ny paralela en OpenCL con threads por bloque no múltiplo de 32')
    ax.grid()

    ax.legend()
    fig.savefig("images/CPU_OpenCL_comparison_min.png")
    plt.show()

    # Zoom
    n_size1 = n_size1[:len(size1) // 2]
    n_size2 = n_size2[:len(size2) // 2]
    n_size3 = n_size3[:len(size3) // 2]
    n_size4 = n_size4[:len(size4) // 2]
    n_eval1 = n_eval1[:len(size1) // 2]
    n_eval2 = n_eval2[:len(size2) // 2]
    n_eval3 = n_eval2[:len(size3) // 2]
    n_eval4 = n_eval4[:len(size4) // 2]

    fig, ax = plt.subplots(figsize=(10,7))
    
    ax.set_xscale('log')

    line1 = ax.plot(n_size1, n_eval1, 'r-o', label='CPU')
    line2 = ax.plot(n_size2, n_eval2, 'b-o', label='8 tpb')
    line3 = ax.plot(n_size3, n_eval3, 'g-o', label='16 tpb')
    line3 = ax.plot(n_size4, n_eval4, 'y-o', label='32 tpb')

    ax.set(xlabel='Tamaño del mundo [Células]', ylabel='Células evaluadas por segundo [Millones]',
       title='Comparación de células evaluadas por segundo entre implementaciones serial en CPU\ny paralela en OpenCL con threads por bloque no múltiplo de 32')
    ax.grid()

    ax.legend()
    fig.savefig("images/CPU_OpenCL_comparison_min_zoom.png")
    plt.show()