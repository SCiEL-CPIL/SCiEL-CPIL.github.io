N = 40

def construct():
    # Hamiltonian cycle: 0-1-2-...-39-0
    edges = set()
    for i in range(N):
        edges.add((min(i, (i+1)%N), max(i, (i+1)%N)))
    
    # Chords (0,k) for k=2..19 to get cycles of length 3..21 and 38..21
    for k in range(2, 20):
        edges.add((0, k))
    
    # Chord (1,39) to get 39-cycle
    edges.add((1, 39))
    
    edges = list(edges)
    
    # Build cycles
    cycles = []
    # Length 3: 0-1-2-0
    cycles.append([0, 1, 2])
    # Lengths 4..20: 0-1-2-...-(k-1)-0
    for k in range(4, 21):
        cycles.append(list(range(k)))
    # Length 21: 0-1-...-20-0
    cycles.append(list(range(21)))
    # Lengths 22..38: 0-k-(k+1)-...-39-0 where k = 40-length
    for length in range(22, 39):
        k = 40 - length
        # Cycle: 0, k, k+1, ..., 39
        cycle = [0] + list(range(k, 40))
        cycles.append(cycle)
    # Length 39: 1-2-...-39-1
    cycles.append(list(range(1, 40)))
    # Length 40: 0-1-2-...-39-0
    cycles.append(list(range(40)))
    
    return {'edges': edges, 'cycles': cycles}
