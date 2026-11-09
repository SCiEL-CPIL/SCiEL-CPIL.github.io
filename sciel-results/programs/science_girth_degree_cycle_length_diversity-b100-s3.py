"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_START = None

def construct():
    global _START
    if _START is None:
        _START = time.time()
    DEADLINE = _START + 85  # safety margin before 90s kill

    # Build two disjoint copies of K_{3,3} on 12 vertices.
    # Each K_{3,3} is 3-regular, triangle-free (bipartite), and its simple
    # cycles have length 4 or 6 only. Disconnected graphs are allowed.
    # Resulting distinct cycle lengths = {4, 6} = 2.
    edges = []
    for offset in (0, 6):
        A = [offset, offset + 1, offset + 2]
        B = [offset + 3, offset + 4, offset + 5]
        for a in A:
            for b in B:
                edges.append((a, b))

    # Deduplicate, normalize, sort, and return as list of [u, v] Python ints.
    seen = set()
    out = []
    for u, v in edges:
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        out.append([int(u), int(v)])
    out.sort()

    if time.time() > DEADLINE:
        return out
    return out
# EVOLVE-BLOCK-END
