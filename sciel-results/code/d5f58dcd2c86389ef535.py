N = 40

def construct():
    edges = set()
    # Hamiltonian cycle
    for i in range(N):
        a, b = i, (i + 1) % N
        edges.add((min(a,b), max(a,b)))
    # Chords from 0 to k for k=2..20
    for k in range(2, 21):
        edges.add((0, k))
    
    edge_list = sorted(edges)
    
    cycles = []
    for L in range(3, 41):  # lengths 3, 4, ..., 40
        if L == 40:
            cyc = list(range(40))
        elif L <= 21:
            cyc = list(range(L))
        else:
            k = 41 - L
            cyc = [0] + list(range(39, k-1, -1))
        cycles.append(cyc)
    
    return {'edges': edge_list, 'cycles': cycles}
