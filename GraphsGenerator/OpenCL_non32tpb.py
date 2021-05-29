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

    size3 = []
    time9 = []
    time27 = []
    time81 = []
    time243 = []
    time729 = []

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

    # Valores potencias de 3

    with open(path + 'opencl_9tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            size3 += [ int(row['width']) * int(row['height']) ]
            tmp_time = float(row['time']) * 0.001
            time9 += [tmp_time / float(row['iter']) ]
            print(size3[-1], time9[-1])

    with open(path + 'opencl_27tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            tmp_size = int(row['width']) * int(row['height']) 
            tmp_time = float(row['time']) * 0.001
            time27 += [tmp_time / float(row['iter']) ]
            print(tmp_size, time27[-1])

    with open(path + 'opencl_81tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            tmp_size = int(row['width']) * int(row['height']) 
            tmp_time = float(row['time']) * 0.001
            time81 += [tmp_time / float(row['iter']) ]
            print(tmp_size, time81[-1])

    with open(path + 'opencl_243tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            tmp_size = int(row['width']) * int(row['height']) 
            tmp_time = float(row['time']) * 0.001
            time243 += [tmp_time / float(row['iter']) ]
            print(tmp_size, time243[-1])

    with open(path + 'opencl_729tpb.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            tmp_size = int(row['width']) * int(row['height']) 
            tmp_time = float(row['time']) * 0.001
            time729 += [tmp_time / float(row['iter']) ]
            print(tmp_size, time729[-1])

    n_size = np.array(size)
    n_time32 = np.array(time32) 
    n_time64 = np.array(time64) 
    n_time128 = np.array(time128) 
    n_time256 = np.array(time256) 
    n_time512 = np.array(time512)  
    n_time1024 = np.array(time1024)  

    n_size3 = np.array(size3)
    n_time9 = np.array(time9) 
    n_time27 = np.array(time27) 
    n_time81 = np.array(time81) 
    n_time243 = np.array(time243) 
    n_time729 = np.array(time729)

    n_eval32 = n_size / n_time32 / 1000000
    n_eval64 = n_size / n_time64 / 1000000
    n_eval128 = n_size / n_time128 / 1000000
    n_eval256 = n_size / n_time256 / 1000000
    n_eval512 = n_size / n_time512 / 1000000
    n_eval1024 = n_size / n_time1024 / 1000000

    n_eval9 = n_size3 / n_time9 / 1000000
    n_eval27 = n_size3 / n_time27 / 1000000
    n_eval81 = n_size3 / n_time81 / 1000000
    n_eval243 = n_size3 / n_time243 / 1000000
    n_eval729 = n_size3 / n_time729 / 1000000

    fig, ax = plt.subplots(figsize=(10,7))
    
    ax.set_xscale('log')

    line32 = ax.plot(n_size, n_eval32, color='#fabed4', marker='^', label='32 tpb')
    line64 = ax.plot(n_size, n_eval64, color='#ffd8b1', marker='^', label='64 tpb')
    line128 = ax.plot(n_size, n_eval128, color='#a9a9a9', marker='^', label='128 tpb')
    line256 = ax.plot(n_size, n_eval256, color='#aaffc3', marker='^', label='256 tpb')
    line512 = ax.plot(n_size, n_eval512, color='#dcbeff', marker='^', label='512 tpb')
    line1024 = ax.plot(n_size, n_eval1024, color='#ffe119',marker='^', label='1024 tpb')

    line9 = ax.plot(n_size3, n_eval9, color='#f032e6', marker='o', label='9 tpb')
    line27 = ax.plot(n_size3, n_eval27, color='#469990', marker='o', label='27 tpb')
    line81 = ax.plot(n_size3, n_eval81, color='#000075', marker='o', label='81 tpb')
    line243 = ax.plot(n_size3, n_eval243, color='#9A6324', marker='o', label='243 tpb')
    line729 = ax.plot(n_size3, n_eval729, color='#e6194B', marker='o', label='729 tpb')

    ax.set(xlabel='Tamaño del mundo [Células]', ylabel='Células evaluadas por segundo [Millones]',
       title='Evaluación del programa en GPU con diferentes threads por bloque (tpb) con OpenCL')
    ax.grid()

    ax.legend()
    fig.savefig("images/OpenCL_non32tpb.png")
    plt.show()