import time

def construct():
    N = 40
    start_time = time.time()
    deadline = 60.0  # Safety margin
    
    # 1. Hamiltonian Cycle
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # 2. Chords (0, k) for k in 2..20
    chords = list(range(2, 21))
    for k in chords:
        edges.add((0, k))
    
    # Convert to list of tuples (sorted for consistency if needed, but set is fine)
    edge_list = [tuple(sorted(e)) for e in edges]
    
    # 3. Cycles
    cycles = []
    
    # Length 40
    cycles.append(list(range(N)))
    
    # Lengths 3 to 39
    # We need them in order 3, 4, ..., 40?
    # Task says: "one vertex-list cycle of every length 3..40 in that order"
    # So cycles[0] is length 3, cycles[1] is length 4, ..., cycles[37] is length 40.
    
    # Re-initialize list to be in order 3..40
    cycles = []
    
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cycles.append(list(range(N)))
        elif L <= 21:
            # Covered by chord k = L - 1
            # Cycle: 0-1-...-(L-1)-0
            cycles.append(list(range(L)))
        else:
            # Covered by chord k = 41 - L
            k = 41 - L
            # Cycle: 0-k-(k+1)-...-39-0
            cycle_vertices = [0] + list(range(k, N))
            cycles.append(cycle_vertices)
            
    return {'edges': edge_list, 'cycles': cycles}
