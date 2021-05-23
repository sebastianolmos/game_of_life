"""Grafico con la comparacion de diferentes tpb usando la implementacion con OPENCL"""

import matplotlib.pyplot as plt
import numpy as np
import csv

path = "Data/"

if __name__ == "__main__":
    size = []
    time32 = []
    time64 = []
    time128 = []
    time256 = []
    time512 = []
    time1024 = []

    with open(path + 'opencl_32tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            size += [ int(row['width']) * int(row['height']) ]
            tmp_time = float(row['time']) * 0.001
            time32 += [tmp_time / float(row['iter']) ]
            print(size[-1], time32[-1])

    with open(path + 'opencl_64tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            tmp_size = int(row['width']) * int(row['height']) 
            tmp_time = float(row['time']) * 0.001
            time64 += [tmp_time / float(row['iter']) ]
            print(tmp_size, time64[-1])

    with open(path + 'opencl_128tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            tmp_size = int(row['width']) * int(row['height']) 
            tmp_time = float(row['time']) * 0.001
            time128 += [tmp_time / float(row['iter']) ]
            print(tmp_size, time128[-1])

    with open(path + 'opencl_256tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            tmp_size = int(row['width']) * int(row['height']) 
            tmp_time = float(row['time']) * 0.001
            time256 += [tmp_time / float(row['iter']) ]
            print(tmp_size, time256[-1])

    with open(path + 'opencl_512tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            tmp_size = int(row['width']) * int(row['height']) 
            tmp_time = float(row['time']) * 0.001
            time512 += [tmp_time / float(row['iter']) ]
            print(tmp_size, time512[-1])

    with open(path + 'opencl_1024tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            tmp_size = int(row['width']) * int(row['height']) 
            tmp_time = float(row['time']) * 0.001
            time1024 += [tmp_time / float(row['iter']) ]
            print(tmp_size, time1024[-1])

    n_size = np.array(size)
    n_time32 = np.array(time32) 
    n_time64 = np.array(time64) 
    n_time128 = np.array(time128) 
    n_time256 = np.array(time256) 
    n_time512 = np.array(time512)  
    n_time1024 = np.array(time1024)  

    n_eval32 = n_size / n_time32
    n_eval64 = n_size / n_time64
    n_eval128 = n_size / n_time128
    n_eval256 = n_size / n_time256
    n_eval512 = n_size / n_time512
    n_eval1024 = n_size / n_time1024

    fig, ax = plt.subplots(figsize=(10,7))
    
    ax.set_xscale('log')

    line32 = ax.plot(n_size, n_eval32, 'b-o', label='32 tbp')
    line64 = ax.plot(n_size, n_eval64, 'r-o', label='64 tbp')
    line128 = ax.plot(n_size, n_eval128, 'y-o', label='128 tbp')
    line256 = ax.plot(n_size, n_eval256, 'g-o', label='256 tbp')
    line512 = ax.plot(n_size, n_eval512, 'm-o', label='512 tbp')
    line1024 = ax.plot(n_size, n_eval1024, 'c-o', label='1024 tbp')

    ax.set(xlabel='Número de celulas', ylabel='Celulas evaluadas por segundo (tpb)',
       title='Evaluacion del programa en GPU con diferentes threads por bloque (tpb) con OpenCL')
    ax.grid()

    ax.legend()
    fig.savefig("images/OpenCL_tpb.png")
    plt.show()