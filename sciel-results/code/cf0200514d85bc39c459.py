import time

N = 40

def construct():
    # Deadline guard
    start_time = time.time()
    
    # Base cycle (rim)
    rim_edges = [(i, (i + 1) % N) for i in range(N)]
    
    # Chords to cover all cycle lengths 3..39
    # Chord (0, k) creates cycles of length k+1 and N-k+1
    # We need lengths 3..39
    # k ranges from 2 to 20
    # k=2 -> 3, 39
    # k=3 -> 4, 38
    # ...
    # k=19 -> 20, 22
    # k=20 -> 21
    
    chords = [(0, k) for k in range(2, 21)]
    
    edges = rim_edges + chords
    
    cycles = []
    
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cyc = list(range(N))
        elif L <= N // 2 + 1:
            # Use chord (0, L-1)
            # Cycle: 0, 1, ..., L-1
            k = L - 1
            cyc = list(range(L))
        else:
            # Use chord (0, k) where N - k + 1 = L => k = N - L + 1
            k = N - L + 1
            # Cycle: 0, k, k+1, ..., N-1
            cyc = [0] + list(range(k, N))
            
        cycles.append(cyc)
        
    return {'edges': edges, 'cycles': cycles}
