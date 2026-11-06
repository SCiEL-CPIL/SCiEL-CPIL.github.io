import time
import sys

N = 40

def construct():
    t0 = time.time()
    deadline = t0 + 80.0

    # Base cycle edges: 0-1-2-...-39-0
    edges = []
    edge_set = set()

    for i in range(N):
        u = i
        v = (i + 1) % N
        e = (min(u, v), max(u, v))
        edges.append(e)
        edge_set.add(e)

    # Add chords from vertex 0 to vertices 2..20
    for d in range(2, 21):
        e = (0, d)
        edges.append(e)
        edge_set.add(e)

    # Generate cycles for lengths 3..40
    cycles = []
    for k in range(3, N + 1):
        if time.time() > deadline:
            break
        if k == N:
            cycle = list(range(N))
        else:
            # Forward: 0, 1, 2, ..., k-1, 0 (needs edge (0, k-1))
            d1 = k - 1
            if (0, d1) in edge_set:
                cycle = list(range(d1 + 1))
            else:
                # Backward: 0, N-1, N-2, ..., d2, 0 (needs edge (0, d2))
                d2 = N - (k - 1)
                if (0, d2) in edge_set:
                    cycle = [0] + list(range(N - 1, d2 - 1, -1))
                else:
                    cycle = list(range(d1 + 1))
        cycles.append(cycle)

    return {'edges': edges, 'cycles': cycles}
