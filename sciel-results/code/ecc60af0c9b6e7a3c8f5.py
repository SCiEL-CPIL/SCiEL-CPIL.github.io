"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    # Vertices 0..39
    # Edges:
    # 1. Path 0-1-...-39 (39 edges)
    # 2. Chords (0, k) for k in 2..37 (36 edges)
    # 3. Chord (1, 39) (1 edge)
    # Total edges: 39 + 36 + 1 = 76
    
    edges = []
    # Path edges
    for i in range(N - 1):
        edges.append((i, i + 1))
        
    # Chords from 0 to 2..37
    for k in range(2, 38): # 2 to 37 inclusive
        edges.append((0, k))
        
    # Chord (1, 39)
    edges.append((1, 39))
    
    # Cycles
    cycles = []
    
    # Lengths 3 to 38
    for k in range(3, 39):
        # Cycle: 0-1-...-(k-1)-0
        cycle = list(range(k))
        cycles.append(cycle)
        
    # Length 39
    # Cycle: 1-2-...-39-1
    cycle_39 = list(range(1, 40))
    cycles.append(cycle_39)
    
    # Length 40
    # Cycle: 0-1-39-38-...-2-0
    # Vertices: 0, 1, 39, 38, ..., 2
    cycle_40 = [0, 1] + list(range(39, 1, -1))
    cycles.append(cycle_40)
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
