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
    # This is the theoretical minimum for a graph with min degree >= 3 and girth >= 4.
    
    edges = []

    # Component 1: K_{3,3} on vertices 0..5
    # Partition A: {0, 1, 2}
    # Partition B: {3, 4, 5}
    for u in range(0, 3):
        for v in range(3, 6):
            edges.append([u, v])

    # Component 2: K_{3,3} on vertices 6..11
    # Partition A: {6, 7, 8}
    # Partition B: {9, 10, 11}
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([u, v])

    # Sort for determinism
    edges.sort()

    # Return as list of [u, v] lists with int types
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])

    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Should never happen for this static construction
        pass

    return result
# EVOLVE-BLOCK-END
