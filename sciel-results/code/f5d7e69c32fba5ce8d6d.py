import time
import random

N = 40

def construct():
    # Deadline guard: 90s limit, use 75s for safety
    deadline = time.time() + 75

    # 1. Base Cycle Edges (Hamiltonian cycle 0-1-...-39-0)
    edges = set()
    for i in range(N):
        j = (i + 1) % N
        edges.add((min(i, j), max(i, j)))

    # 2. Add Chords
    # To cover cycle lengths 3..39, we need chords.
    # A chord (u, v) creates cycles of length (dist(u,v) on base cycle) + 1
    # and (N - dist(u,v)) + 1.
    # Using hub 0, chords (0, k) for k=2..20 cover all lengths:
    # L=3 -> k=2 (dist 2, cycle 0-1-2-0)
    # L=4 -> k=3 (dist 3, cycle 0-1-2-3-0)
    # ...
    # L=21 -> k=20 (dist 20, cycle 0-1-...-20-0)
    # L=22 -> k=19 (dist 21, cycle 0-19-20-...-39-0, length 21+1=22)
    # ...
    # L=39 -> k=2 (dist 38, cycle 0-2-3-...-39-0, length 38+1=39)

    chords_k = list(range(2, N // 2 + 1))  # 2..20

    for k in chords_k:
        e = (0, k)
        edges.add(e)

    edge_list = [tuple(sorted(e)) for e in edges]

    # 3. Construct Cycles
    cycles = []

    for L in range(3, N + 1):
        if L == N:
            cycle_verts = list(range(N))
        else:
            # Try to use chord (0, L-1)
            k1 = L - 1
            k2 = N + 1 - L

            # We added chords for k in 2..20.
            # If k1 is in 2..20, use it.
            if 2 <= k1 <= N // 2:
                # Cycle: 0 -> 1 -> ... -> k1 -> 0
                # Vertices: 0, 1, ..., k1
                # Length: k1 - 0 + 1 (path) + 1 (chord) = k1 + 1 = L.
                cycle_verts = list(range(k1 + 1))
            else:
                # k1 > 20. Then k2 < 20.
                # Also k2 >= 2 because L <= N-1 = 39 => k2 = 41-L >= 2.
                # So k2 is in 2..19, which is in our chord set.
                # Use chord (0, k2).
                # Cycle: 0 -> k2 -> k2+1 -> ... -> N-1 -> 0
                # Vertices: 0, k2, k2+1, ..., N-1
                # Path length: N-1 - k2 + 1 = N - k2.
                # Plus chord: N - k2 + 1 = (40 - k2) + 1 = 41 - k2 = L.
                cycle_verts = [0] + list(range(k2, N))

        cycles.append(cycle_verts)

    return {'edges': edge_list, 'cycles': cycles}
