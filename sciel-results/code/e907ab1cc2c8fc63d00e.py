import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 60.0
    
    edges = set()
    
    # Base Hamiltonian cycle: 0-1-2-...-39-0
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # Chords from 0 to k for k in 2..20
    # This generates cycles of length k+1 and 40-k+1
    # k=2 -> 3, 39
    # k=3 -> 4, 38
    # ...
    # k=20 -> 21, 21
    for k in range(2, 21):
        u, v = 0, k
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    edge_list = sorted(list(edges))
    
    # Generate cycles
    # We need cycles of length 3 to 40 in that order
    cycle_by_length = {40: list(range(N))}
    
    for k in range(2, 21):
        # Cycle 1: 0 -> 1 -> ... -> k -> 0
        len1 = k + 1
        cycle1 = list(range(k + 1))
        cycle_by_length[len1] = cycle1
        
        # Cycle 2: 0 -> 39 -> 38 -> ... -> k -> 0
        len2 = N - k + 1
        # Path from 0 to k going "backwards" via 39
        # 0, 39, 38, ..., k
        cycle2 = [0] + list(range(N - 1, k - 1, -1))
        cycle_by_length[len2] = cycle2

    # Build the ordered list for lengths 3..40
    final_cycles = [cycle_by_length[L] for L in range(3, N + 1)]
    
    return {'edges': edge_list, 'cycles': final_cycles}
