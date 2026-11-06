"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    # Hamiltonian cycle edges
    edges = [(i, (i + 1) % N) for i in range(N)]
    # Chords to ensure pancyclic property
    # Chord (0, k) creates cycles of length k+1 and N-k+1
    # We need to cover lengths 3 to N-1
    # k ranges from 2 to N//2
    for k in range(2, N // 2 + 1):
        edges.append((0, k))
    
    # Generate cycles
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cycle = list(range(N))
        else:
            # Find k such that L = k+1 or L = N-k+1
            # k = L-1
            k = L - 1
            if k <= N // 2:
                # Use forward path 0->1->...->k
                cycle = list(range(k + 1))
            else:
                # Use backward path 0->N-1->...->N-k
                # N-k = N - (L-1)
                end_vertex = N - k
                # Path is 0, N-1, N-2, ..., end_vertex
                # Length of path is N - end_vertex = N - (N-k) = k
                # Cycle length k+1 = L
                cycle = [0] + list(range(N - 1, end_vertex - 1, -1))
        cycles.append(cycle)
        
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
