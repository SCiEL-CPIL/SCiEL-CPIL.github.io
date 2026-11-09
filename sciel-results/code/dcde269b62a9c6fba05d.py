"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a graph on 12 vertices (0..11) consisting of two disjoint K_{3,3} components.
    This graph is 3-regular, triangle-free (bipartite), and contains only cycles of length 4 and 6.
    The number of distinct cycle lengths is 2, which is the theoretical minimum
    for a graph with min degree 3 and girth >= 4.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit

    edges = set()

    # First K_{3,3} on vertices 0..5
    # Partition A: 0, 1, 2
    # Partition B: 3, 4, 5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.add((u, v) if u < v else (v, u))

    # Second K_{3,3} on vertices 6..11
    # Partition A: 6, 7, 8
    # Partition B: 9, 10, 11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.add((u, v) if u < v else (v, u))

    # Convert to list of lists with proper int types
    result = [[int(u), int(v)] for (u, v) in sorted(edges)]

    return result
# EVOLVE-BLOCK-END
