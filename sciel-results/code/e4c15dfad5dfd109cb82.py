import time
import sys

N = 40

def construct():
    start_time = time.time()
    
    # Optimal Construction: Hamiltonian Cycle + 19 Chords
    # Total Edges: 59
    # Score: 19
    
    edges = set()
    
    # 1. Base Hamiltonian Cycle
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # 2. Chords from 0 to k for k in 2..20
    # Covers lengths:
    # k=2 -> 3, 39
    # k=3 -> 4, 38
    # ...
    # k=19 -> 20, 22
    # k=20 -> 21
    for k in range(2, 21):
        u, v = 0, k
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # 3. Generate Cycles
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycle_verts = list(range(N))
        else:
            if L <= 21:
                # Short path: 0 -> 1 -> ... -> L-1 -> 0
                # Chord (0, L-1)
                k = L - 1
                cycle_verts = list(range(k + 1))
            else:
                # Long path: 0 -> N-1 -> ... -> k -> 0
                # Length L = (N - k) + 1 => k = N - L + 1
                k = N - L + 1
                cycle_verts = [0] + list(range(N - 1, k - 1, -1))
        cycles.append(cycle_verts)
        
    # Return the result
    return {
        'edges': [tuple(sorted(e)) for e in edges],
        'cycles': cycles
    }
