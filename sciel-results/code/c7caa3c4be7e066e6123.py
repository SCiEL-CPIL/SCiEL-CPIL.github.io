"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 60.0  # safety margin before 90s kill

    # Strategy: Construct two disjoint K_{3,3} components.
    # K_{3,3} is 3-regular, bipartite (girth 4), and has cycles of length 4 and 6 only.
    # Two disjoint copies ensure the minimum degree is 3 for all 12 vertices.
    # The set of distinct cycle lengths in the union is {4, 6}.
    # Raw value = 2.
    # Score = 5.0 / 2.0 = 2.5.
    
    edges = []

    # Component 1: K_{3,3} on vertices 0..5
    # Partition A: {0, 1, 2}
    # Partition B: {3, 4, 5}
    for u in (0, 1, 2):
        for v in (3, 4, 5):
            edges.append([u, v])

    # Component 2: K_{3,3} on vertices 6..11
    # Partition A: {6, 7, 8}
    # Partition B: {9, 10, 11}
    for u in (6, 7, 8):
        for v in (9, 10, 11):
            edges.append([u, v])

    # Sort for determinism
    edges.sort()

    # Return as list of [u, v] lists with int types
    result = [[int(e[0]), int(e[1])] for e in edges]

    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Should never happen for this static construction, but guard anyway
        pass

    return result
# EVOLVE-BLOCK-END
