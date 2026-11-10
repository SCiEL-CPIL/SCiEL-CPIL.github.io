N = 40

def construct():
    edges = []
    # C40: edges (i, (i+1)%N)
    for i in range(N):
        edges.append((i, (i+1) % N))
    # Chords (0, k) for k = 2..20
    for k in range(2, 21):
        edges.append((0, k))
    
    # Cycles
    cycles = []
    for length in range(3, N+1):
        if length == N:
            cycles.append(list(range(N)))
        elif length <= 21:
            # Short side: 0, 1, 2, ..., length-1, 0
            k = length - 1
            cycles.append(list(range(0, k+1)))
        else:
            # Long side: 0, 39, 38, ..., N-length+1, 0
            # Length = 41-k where k is the chord endpoint
            # We need cycle of given length L > 21
            # L = 41 - k => k = 41 - L
            k = 41 - length
            # Cycle: 0, 39, 38, ..., k, 0
            # That's vertices: 0, 39, 38, ..., k
            # Number of vertices: 1 + (39 - k + 1) = 40 - k = 40 - (41-L) = L-1+1 = L. Good.
            cyc = [0] + list(range(39, k-1, -1))
            cycles.append(cyc)
    
    return {'edges': edges, 'cycles': cycles}
