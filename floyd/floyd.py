
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from collections import deque
import time
import random

cCode = """

    __device__ float minm(float a, float b) {
        return (a < b) ? a : b;
    }

    __global__ void floyd(float *dist, int *size) {
        int i = threadIdx.x+blockIdx.x*blockDim.x;
        int side = size[0];
        __syncthreads();
        for (int k = 0; k < side; k++) {
            for (int j = 0; j < side; j++) {
                int ij = i*side+j, ik = i*side+k, kj = k*side+j;
                //printf("I: %d ==> ij: %d, ik: %d, kj: %d\\n", i, ij, ik, kj);
                //printf("%d, %d\\n", dist[ij], dist[ik]+dist[kj]);
                dist[ij] = minm(dist[ij], dist[ik]+dist[kj]);
            }
            __syncthreads();
        }
    }

"""

kernel = SourceModule(cCode)
floyd = kernel.get_function("floyd");

def GtoDist(G):
    dist = []
    for u in G:
        for v in u:
            dist.append(v)
    return dist


def GPU(G, higher):
    side = len(G)
    dist = GtoDist(G)

    tmpbegin = time.perf_counter()

    dist_gpu = gpuarray.to_gpu(np.array(dist, dtype=np.float32))
    size_gpu = gpuarray.to_gpu(np.array([side], dtype=np.int32))


    floyd(dist_gpu, size_gpu, block=(side,1,1), grid=(1,1,1))
    tmpend = time.perf_counter()
    timed = (tmpend-tmpbegin)

    temp = np.array([0 for _ in range(side*side)], dtype=np.float32)
    dist_gpu.get(temp)

    result = []
    for u in range(side):
        tmp = []
        for v in range(side):
            diff = max(temp[u*side+v],higher)-min(temp[u*side+v],higher) > 0.000001
            tmp.append(temp[u*side+v] if diff else "I")
        result.append(tmp)

    print("|||||||||| GPU ||||||||||")
    print("Time:",timed)
    #print("Result:",result)
    return result, timed


def CPU(G, higher):
    dist = G.copy()
    tmpbegin = time.perf_counter()
    for k in range(len(dist)):
        for i in range(len(dist)):
            for j in range(len(dist)):
                #print(dist[i][j], dist[i][k]+dist[k][j])
                dist[i][j] = min(dist[i][j], dist[i][k]+dist[k][j])
    for u in range(len(dist)):
        for v in range(len(dist)):
            if dist[u][v] == higher:
                dist[u][v] = 'I'
    tmpend = time.perf_counter()
    timed = (tmpend-tmpbegin)
    print("|||||||||| CPU ||||||||||")
    print("Time:",timed)
    #print("Result:",dist)
    return dist, timed


def comparison(size):
    def generate(size):
        G = [[] for _ in range(size)]
        higher = -1
        for u in range(size):
            for v in range(size):
                if u != v:
                    num = random.randint(1, 20)
                    higher = max(num, higher)
                    G[u].append(float(num))
                else:
                    G[u].append(0.0)
        f = open("comparisonGraph.txt", 'w')
        f.write(str(G))
        f.close()
        return G, higher

    G, higher = generate(size)
    higher += 1
    gpuDist, gpuTime = GPU(G.copy(), higher)
    cpuDist, cpuTime = CPU(G.copy(), higher)
    print("Results are equal:", gpuDist==cpuDist)
    print("SpeedUp:", cpuTime/gpuTime)
    return
