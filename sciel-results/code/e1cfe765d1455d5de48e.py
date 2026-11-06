"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    edges = set()
    
    # Base Hamiltonian cycle: 0-1-2-...-39-0
    for i in range(N):
        edges.add((min(i, (i+1) % N), max(i, (i+1) % N)))
    
    # Add chords from vertex 0 to vertices 2, 3, ..., 38
    # This allows forming k-cycles: 0-1-2-...-(k-1)-0 for k=3..39
    for k in range(3, N):  # k is the cycle length, need chord (0, k-1)
        v = k - 1
        if v != 0 and v != 1 and v != N-1:  # (0,1) and (0,39) already in base cycle
            edges.add((0, v))
    
    edges_list = sorted(list(edges))
    
    # Build cycles for each length 3..40
    cycles = []
    for k in range(3, N+1):
        if k == N:
            # The full Hamiltonian cycle
            cycle = list(range(N))
        else:
            # Cycle: 0-1-2-...-(k-1)-0
            cycle = list(range(k))
        cycles.append(cycle)
    
    return {'edges': edges_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
