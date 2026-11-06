import time

N = 40

def construct():
    start = time.time()
    
    # Base cycle edges
    edges = set()
    for i in range(N):
        a, b = i, (i + 1) % N
        edges.add((min(a, b), max(a, b)))
    
    # Chords from 0 to even vertices 2..38
    for v in range(2, N, 2):
        edges.add((0, v))
    
    edge_list = sorted(edges)
    edge_set = set()
    for a, b in edge_list:
        edge_set.add((a, b))
        edge_set.add((b, a))
    
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycle = list(range(N))
        elif L % 2 == 1:
            # L = 2j+1, j=(L-1)//2, chord (0, 2j)
            j = (L - 1) // 2
            cycle = list(range(2 * j + 1))  # 0,1,...,2j
        else:
            # L = 2j, chord (0, 40-L), cycle 0,39,38,...,40-L
            # Need L vertices: 0 and L-1 vertices from 39 down
            end = N - L
            cycle = [0] + list(range(N - 1, end - 1, -1))
        
        cycles.append(cycle)
    
    # Validate
    for cycle in cycles:
        if len(set(cycle)) != len(cycle):
            continue
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            if (u, v) not in edge_set:
                return {'edges': edge_list, 'cycles': cycles}
    
    return {'edges': edge_list, 'cycles': cycles}
