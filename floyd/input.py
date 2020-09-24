
import csv

def getData(path):
    result = []
    with open(path.replace('\\', '/'), "r") as file:
        reader = csv.reader(file)
        print("Columns")
        s, e, w = int(input("  Start Node: ")), int(input("  End Node: ")), int(input("  Weight: "))
        for row in reader:
            result.append([int(row[s]), int(row[e]), float(row[w])/1000])
    return result, max([max(n[0], n[1]) for n in result]), max([n[2] for n in result])+1

def toMatrix(edges, size, higher):
    print(size, flush=True)
    G = [[higher for _ in range(size+1)] for _ in range(size+1)]
    for edge in edges:
        u, v, w = edge
        G[u][v] = w
    return G

def toDivideMatrix(edges, limit, maximum, higher):
    quantity = maximum//limit
    Gs = []
    for n in range(quantity):
        Gs.append([[higher for v in range(limit*n, limit*(n+1))] for u in range(limit*n, limit*(n+1))])
    Gs.append([[higher for u in range(quantity*limit, maximum+1)] for n in range(quantity*limit, maximum+1)])
    for u,v,w in edges:
        if (u//limit == v//limit):
            m = u//limit
            u = u%limit
            v = v%limit
            Gs[m][u][v] = w
    return Gs

def choose(limit):
    data, size, higher = getData(input("Absolute path to Data Set CSV: "))
    if size > limit:
        Gs = toDivideMatrix(data, limit, size, higher)
    else:
        Gs = [toMatrix(data, size, higher)]
    return Gs, higher
