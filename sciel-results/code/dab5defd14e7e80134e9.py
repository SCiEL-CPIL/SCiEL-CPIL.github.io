N = 40

# EVOLVE-BLOCK-START
def construct():
    edges = []
    cycles = []
    
    # 1. Base Cycle Edges: (i, i+1) mod N
    for i in range(N):
        edges.append((i, (i + 1) % N))
        
    # 2. Chord Edges from 0 to k for k in 2..20
    # These chords allow generating cycles of length k+1 and N-k+1
    for k in range(2, 21):
        edges.append((0, k))
        
    # 3. Construct Cycles
    # Lengths 3 to 21: [0, 1, ..., L-1]
    for L in range(3, 22):
        cycles.append(list(range(L)))
        
    # Lengths 22 to 39: [0, k, k+1, ..., N-1] where L = N - k + 1 => k = N - L + 1
    for L in range(22, 40):
        k = N - L + 1
        # Vertices: 0, k, k+1, ..., N-1
        cyc_vertices = [0] + list(range(k, N))
        cycles.append(cyc_vertices)
        
    # Length 40: [0, 1, ..., N-1]
    cycles.append(list(range(N)))
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
