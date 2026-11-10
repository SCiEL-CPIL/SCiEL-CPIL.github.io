"""Sparse pancyclic graph on 40 vertices: cycle + 12 chords from vertex 0."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    # Base: Hamiltonian cycle 0-1-2-...-39-0
    edges = [(i, (i + 1) % N) for i in range(N)]

    # Chords from vertex 0; each chord (0, j) adds cycles of length j+1 and N-j+1
    chords = [2, 3, 4, 5, 6, 8, 9, 11, 13, 15, 17, 19]
    for j in chords:
        edges.append((0, j))

    # Build cycle certificates for every length L in 3..N (in that order)
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Full Hamiltonian cycle
            cyc = list(range(N))
        elif L <= N // 2 + 1:
            # Short arc 0-1-...-(L-1)-0  (uses chord (0, L-1))
            cyc = list(range(L))
        else:
            # Long arc 0-(N-L+1)-...-N-1-0  (uses chord (0, N-L+1))
            j = N - L + 1
            cyc = [0] + list(range(j, N))
        cycles.append(cyc)

    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
