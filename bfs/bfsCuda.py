
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from collections import deque
import time
import random

cCode = """

    __global__ void bfsKernel(int *VS, int *VL, int *E, bool *S, bool *visited, int *dist, bool *done) {
      int id = threadIdx.x;// + (blockIdx.x * blockDim.x);
      if (S[id] && !visited[id]) {
        S[id] = false;
        visited[id] = true;
        __syncthreads();
        int i;
        for (i = VS[id]; i < VS[id]+VL[id]; i++) {
          if (!visited[E[i]]) {
            dist[E[i]] = dist[id]+1;
            S[E[i]] = true;
            done[0] = false;
          }
        }
      }
    }

"""

kernel = SourceModule(cCode)
bfs = kernel.get_function("bfsKernel");


def adjacencyListToFormat(G):
    vs, vl, e, acum = [], [], [], 0
    for u in G:
        vs.append(acum)
        vl.append(len(u))
        acum += len(u) or 1
        for v in u:
            e.append(v)
    return (len(G), vs, vl, e)

def shortestGPU(G, source):
    nodes, vs, vl, e = adjacencyListToFormat(G)

    VS_gpu = gpuarray.to_gpu(np.array(vs, dtype=np.int32))
    VL_gpu = gpuarray.to_gpu(np.array(vl, dtype=np.int32))
    E_gpu = gpuarray.to_gpu(np.array(e, dtype=np.int32))
    S_gpu = gpuarray.to_gpu(np.array([False if n != source else True for n in range(nodes)], dtype=np.bool_))
    visited_gpu = gpuarray.to_gpu(np.array([False for _ in range(nodes)], dtype=np.bool_))
    dist_gpu = gpuarray.to_gpu(np.array([0 for _ in range(nodes)], dtype=np.int32))
    done_gpu = gpuarray.to_gpu(np.array([True], dtype=np.bool_))
    size_gpu = gpuarray.to_gpu(np.array([nodes], dtype=np.int32))

    done = np.array([False], dtype=np.bool_)

    tmpbegin = time.perf_counter()
    while (not done[0]):
        done_gpu = gpuarray.to_gpu(np.array([True], dtype=np.bool_))
        bfs(VS_gpu, VL_gpu, E_gpu, S_gpu, visited_gpu, dist_gpu, done_gpu, block=(nodes,1,1), grid=(1,1,1))
        done_gpu.get(done)
    tmpend = time.perf_counter()

    timed = (tmpend-tmpbegin)

    print("|||| GPU Execution: ||||")
    print("Time: ", timed)
    print("Visited nodes: ", visited_gpu)
    result = np.array([0 for _ in range(nodes)], dtype=np.int32)
    dist_gpu.get(result)
    result = [float("inf") if not result[r] and r != source else result[r] for r in range(len(result))]
    print("Shortest distances from source: ", result)
    return (result, timed)

def shortestCPU(G, source):
    dist = [float("inf") for _ in range(len(G))]
    visited = [False for _ in range(len(G))]
    dq = deque([(source, 0)])
    dist[source] = 0
    begin = time.perf_counter()
    while len(dq):
        u, d = dq.popleft()
        if not visited[u]:
            visited[u] = True
            for v in G[u]:
                dist[v] = min(dist[v], d+1)
                if not visited[v]:
                    dq.append((v, d+1))
    end = time.perf_counter()
    print("|||| CPU Execution: ||||")
    print("Time: ", end-begin)
    print("Visited nodes: ", visited)
    print("Shortest distances from source: ", dist)
    return (dist, end-begin)

def generate(size):
    G = [[] for _ in range(size)]
    for u in range(size):
        for v in range(size):
            if u != v:
                num = random.randint(0, 10)
                if num >= 6:
                    G[u].append(v)
    f = open("graph.txt", 'w')
    f.write(str(G))
    f.close()
    return G

def equal(ar1, ar2):
    result = True
    if len(ar1) == len(ar2):
        for u in range(len(ar1)):
            if ar1[u] != ar2[u]:
                result = False
    else:
        result = False
    return result


def main():
    G = generate(int(input()))
    gpuDist, gpuTime = shortestGPU(G, 0)
    print()
    cpuDist, cpuTime = shortestCPU(G, 0)
    print()
    print(len(G))
    print("Are the results equal? ", equal(gpuDist, cpuDist))

    print("SpeadUp: ", cpuTime/gpuTime)


main()
