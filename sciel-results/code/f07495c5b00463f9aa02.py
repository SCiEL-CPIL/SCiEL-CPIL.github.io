N = 40

def construct():
    edges = set()
    # Hamiltonian cycle
    for i in range(N):
        j = (i + 1) % N
        edges.add((min(i,j), max(i,j)))
    
    # Chords from 0 to 2,3,...,21
    for d in range(2, 22):
        edges.add((0, d))
    
    edges = list(edges)
    
    cycles = []
    for L in range(3, N+1):
        if L <= 22:
            # L = d+1, d = L-1
            d = L - 1
            cycle = list(range(0, d+1))  # 0,1,2,...,d
        else:
            # L = 41-d, d = 41-L
            d = 41 - L
            cycle = [0] + list(range(39, d-1, -1))  # 0, 39, 38, ..., d
        cycles.append(cycle)
    
    return {'edges': edges, 'cycles': cycles}
