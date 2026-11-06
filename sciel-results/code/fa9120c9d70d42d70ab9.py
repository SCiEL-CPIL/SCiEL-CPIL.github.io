"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    # Construct a fan graph:
    # Vertex 0 is connected to all other vertices 1..N-1.
    # Vertices 1..N-1 form a path.
    # This ensures cycles of all lengths 3..N exist.
    # Cycle of length k: 0 -> 1 -> 2 -> ... -> k-1 -> 0
    edges = []
    for i in range(1, N):
        edges.append((0, i))
    for i in range(1, N - 1):
        edges.append((i, i + 1))
    
    cycles = []
    for k in range(3, N + 1):
        # Cycle 0-1-...-(k-1)-0
        cycle = list(range(k))
        cycles.append(cycle)
        
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
