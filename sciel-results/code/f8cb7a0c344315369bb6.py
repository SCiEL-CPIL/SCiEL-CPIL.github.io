import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Base cycle edges
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # Chords (0, j) for j in 2..20
    # This covers lengths 3..21 (short) and 39..21 (long)
    # Wait, j=20 gives short 21, long 21.
    # j=2 gives short 3, long 39.
    # So j in 2..20 covers 3..21 and 39..21.
    # Union is 3..39.
    # Plus base cycle 40.
    chords = []
    for j in range(2, 21):
        edges.add((0, j))
        chords.append(j)
    
    # Convert to list of tuples for output
    edge_list = list(edges)
    
    # Construct cycles
    cycles = []
    
    # Length 3 to 39
    for L in range(3, N):
        if L <= 21:
            # Use chord (0, L-1)
            # Cycle: 0, 1, ..., L-1
            cycle = list(range(L))
            # Ensure it's a cycle: 0-1-...-(L-1)-0
            # The list of vertices is 0, 1, ..., L-1
            cycles.append(cycle)
        else:
            # L > 21
            # Use chord (0, 41-L)
            j = 41 - L
            # Cycle: 0, 39, 38, ..., j
            # Vertices: 0, then N-1 down to j
            cycle = [0] + list(range(N - 1, j - 1, -1))
            cycles.append(cycle)
            
    # Length 40
    # Cycle: 0, 1, ..., 39
    cycles.append(list(range(N)))
    
    return {'edges': edge_list, 'cycles': cycles}
