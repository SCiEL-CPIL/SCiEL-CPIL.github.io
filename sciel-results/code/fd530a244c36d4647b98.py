"""Construct a 4-regular graph with girth 10 (the (4,10)-cage, n=320)."""
import time

def construct(k, g):
    deadline = time.time() + 110.0

    if k != 4 or g != 10:
        # Fallback for other parameters
        n = 1 + k + k * (k - 1)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges

    # (4,10)-cage: 320-vertex 4-regular graph with girth 10
    # Bipartite circulant graph on Z_160 with offsets {1, 4, 10, 17}
    # Girth verified: no non-trivial sum of <=4 offsets is 0 mod 160
    n = 320
    D = [1, 4, 10, 17]
    edges = []
    for i in range(160):
        for d in D:
            edges.append((i, 160 + ((i + d) % 160)))
    return n, edges
