N = 40

def construct():
    edges = []
    # Backbone cycle
    for i in range(N):
        edges.append((i, (i + 1) % N))
    
    # Chords from 0 to 2..20
    for k in range(2, 21):
        edges.append((0, k))
        
    # Remove duplicates just in case, though (0,1) and (0,39) are in backbone
    # Backbone has (0,1) and (39,0).
    # Chords are (0,2)..(0,20).
    # No overlap.
    
    edge_set = set()
    for u, v in edges:
        if u > v: u, v = v, u
        edge_set.add((u, v))
    edges = list(edge_set)

    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycles.append(list(range(N)))
        elif L <= 21:
            # Cycle 0-1-...-(L-1)-0
            cycles.append(list(range(L)))
        else:
            # Cycle 0-k-...-39-0
            k = 41 - L
            # Vertices: 0, k, k+1, ..., 39
            cyc = [0] + list(range(k, N))
            cycles.append(cyc)
            
    return {'edges': edges, 'cycles': cycles}
