"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    edges = []
    # 40-cycle: (0,1), (1,2), ..., (38,39), (39,0)
    for i in range(N):
        edges.append((i, (i + 1) % N))
    # Chords (0, 2j) for j = 1..19
    for j in range(1, 20):
        edges.append((0, 2 * j))
    
    cycles = []
    for k in range(3, N + 1):
        if k == N:
            cycles.append(list(range(N)))
        elif k % 2 == 0:
            # Even k: cycle 0 -> k -> k-1 -> ... -> 2 -> 0
            cycles.append([0] + list(range(k, 1, -1)))
        else:
            # Odd k: cycle 0 -> k-1 -> k-2 -> ... -> 1 -> 0
            cycles.append([0] + list(range(k - 1, 0, -1)))
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
