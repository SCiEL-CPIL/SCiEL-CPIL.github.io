import time

N = 40
DEADLINE = 85  # seconds, safety margin before 90s kill

def construct():
    start = time.time()
    
    edges = set()
    cycles = []
    
    # 1. Base Hamiltonian Cycle
    # Edges (i, i+1) mod N
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # 2. Chords from 0 to k for k in 2..20
    # This generates cycles of length k+1 and N-k+1
    # 19 chords is the theoretical minimum: 37 non-Hamiltonian cycle lengths
    # (3..39), each chord covers at most 2, so ceil(37/2)=19.
    for k in range(2, 21):
        edges.add((0, k))
    
    edge_list = list(edges)
    
    # Generate cycles for lengths 3..N
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cycle = list(range(N))
        elif 3 <= L <= 21:
            # L = k + 1 => k = L - 1
            # Cycle: 0-1-...-(L-1)-0
            cycle = list(range(L))
        else:
            # 22 <= L <= 39
            # L = N - k + 1 => k = N + 1 - L
            # Cycle: 0-k-(k+1)-...-(N-1)-0
            k = N + 1 - L
            # Path from k to N-1, then 0
            # Vertices: 0, k, k+1, ..., N-1
            cycle = [0] + list(range(k, N))
            
        cycles.append(cycle)
        
    # Deadline guard (construction is O(N), well within budget)
    if time.time() - start > DEADLINE:
        return {'edges': edge_list, 'cycles': cycles}
    
    return {'edges': edge_list, 'cycles': cycles}
