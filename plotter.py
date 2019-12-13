#!/usr/bin/python3

from collections import defaultdict

import os

results = defaultdict(lambda: defaultdict(float))

directory = 'results'
for file in os.listdir(os.fsencode(directory)):
    data = open(directory + '/' + os.fsdecode(file), 'r').readlines()
    grid_size = 0
    nodes = 0
    time = 0
    for line in data:
        if 'Grid size' in line:
            grid_size = line.split(': ')[1][:-1]
        if 'MPI processes' in line:
            nodes = int(line.split(': ')[1])
        if 'OMP Threads' in line:
            if int(line.split(': ')[1]) == 3:
                grid_size += '-openmp'
        if 'Time' in line:
            time = float(line.split(': ')[1])
    results[grid_size][nodes] = time

print('GridSize', 'NodesAmount', 'TimeMPI', 'AccelerationMPI',
                                 'TimeMPIopenmp', 'AccelerationMPIopenmp', 'MPIopenmp/MPI')
for type, nodes in results.items():
    for nodes_amount, time in sorted(nodes.items()):
        if 'openmp' not in type:
            print(type, nodes_amount, '%.3f' % time, '%.3f' % (results[type][128] / time),
                  '%.3f' % results[type + '-openmp'][nodes_amount],
                  '%.3f' % (results[type + '-openmp'][128] / results[type + '-openmp'][nodes_amount]),
                  '%.3f' % (time / results[type + '-openmp'][nodes_amount]))
