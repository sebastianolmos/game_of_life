"""Grafico con la comparacion de diferentes tpb usando la implementacion con CUDA"""

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

    with open(path + 'cuda_32tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            size += [ int(row['width']) * int(row['height']) ]
            tmp_time = float(row['time']) * 0.001
            time32 += [tmp_time / float(row['iter']) ]
            print(size[-1], time32[-1])

    with open(path + 'cuda_64tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            tmp_size = int(row['width']) * int(row['height']) 
            tmp_time = float(row['time']) * 0.001
            time64 += [tmp_time / float(row['iter']) ]
            print(tmp_size, time64[-1])

    with open(path + 'cuda_128tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            tmp_size = int(row['width']) * int(row['height']) 
            tmp_time = float(row['time']) * 0.001
            time128 += [tmp_time / float(row['iter']) ]
            print(tmp_size, time128[-1])

    with open(path + 'cuda_256tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            tmp_size = int(row['width']) * int(row['height']) 
            tmp_time = float(row['time']) * 0.001
            time256 += [tmp_time / float(row['iter']) ]
            print(tmp_size, time256[-1])

    with open(path + 'cuda_512tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            tmp_size = int(row['width']) * int(row['height']) 
            tmp_time = float(row['time']) * 0.001
            time512 += [tmp_time / float(row['iter']) ]
            print(tmp_size, time512[-1])

    with open(path + 'cuda_1024tpb.csv', mode='r') as csv_file:
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

    n_eval32 = n_size / n_time32 / 1000000
    n_eval64 = n_size / n_time64 / 1000000
    n_eval128 = n_size / n_time128 / 1000000
    n_eval256 = n_size / n_time256 / 1000000
    n_eval512 = n_size / n_time512 / 1000000
    n_eval1024 = n_size / n_time1024 / 1000000

    fig, ax = plt.subplots(figsize=(10,7))
    
    ax.set_xscale('log')

    line32 = ax.plot(n_size, n_eval32, 'b-o', label='32 tpb')
    line64 = ax.plot(n_size, n_eval64, 'r-o', label='64 tpb')
    line128 = ax.plot(n_size, n_eval128, 'y-o', label='128 tpb')
    line256 = ax.plot(n_size, n_eval256, 'g-o', label='256 tpb')
    line512 = ax.plot(n_size, n_eval512, 'm-o', label='512 tpb')
    line1024 = ax.plot(n_size, n_eval1024, 'c-o', label='1024 tpb')

    ax.set(xlabel='Tamaño del mundo [Células]', ylabel='Células evaluadas por segundo [Millones]',
       title='Comparación de células evaluadas por segundo entre implementaciones\ncon diferentes threads por bloque (tpb) para CUDA')
    ax.grid()

    ax.legend()
    fig.savefig("images/CUDA_tpb.png")
    plt.show()