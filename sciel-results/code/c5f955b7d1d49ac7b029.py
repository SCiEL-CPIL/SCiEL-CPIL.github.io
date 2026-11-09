"""Few cycle lengths under degree and girth constraints."""

def construct():
    """
    Construct a graph on vertices 0..11 with:
    - Minimum degree >= 3
    - Girth >= 4 (triangle-free)
    - At least one cycle
    - Minimize the number of distinct simple cycle lengths
    
    Strategy: Use two disjoint K_{3,3} components.
    Each K_{3,3} is bipartite (so triangle-free), 3-regular, and has girth 4.
    The only cycle lengths in K_{3,3} are 4 and 6.
    With two disjoint copies, the set of distinct cycle lengths remains {4, 6}.
    Raw value = 2.
    """
    edges = []
    for offset in (0, 6):
        left = [offset, offset + 1, offset + 2]
        right = [offset + 3, offset + 4, offset + 5]
        for u in left:
            for v in right:
                edges.append([int(u), int(v)])
    return edges
