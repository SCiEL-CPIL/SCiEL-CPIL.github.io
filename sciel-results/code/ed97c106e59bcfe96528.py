def construct():
    N = 40
    
    # 1. Base Cycle Edges
    # Vertices 0..39 in a ring.
    # Edges: (0,1), (1,2), ..., (38,39), (39,0)
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # 2. Chords from Hub 0
    # Add edges (0, k) for k in 2..20
    # Chord (0,k) creates cycle of length k+1 (0->1->...->k->0)
    # and complementary cycle of length N-k+1 (0->k->k+1->...->N-1->0)
    for k in range(2, 21):
        edges.add((0, k))
        
    edge_list = sorted(list(edges))
    
    # 3. Generate Cycles
    # We need cycles of length L for L in 3..40
    cycles = []
    
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cycle_verts = list(range(N))
        elif L <= 21:
            # Cycle: 0->1->2->...->(L-1)->0
            # Uses chord (0, L-1)
            cycle_verts = list(range(L))
        else:
            # Cycle: 0->k->(k+1)->...->(N-1)->0 where k = N+1-L
            # Uses chord (0, k)
            k = N + 1 - L
            cycle_verts = [0] + list(range(k, N))
                
        cycles.append(cycle_verts)
    
    return {'edges': edge_list, 'cycles': cycles}
