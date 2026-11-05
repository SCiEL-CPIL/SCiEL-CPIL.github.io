import time

def construct():
    """
    Returns a simple graph on n vertices (0..n-1) that is:
    - C5-free (no 5-cycle as a subgraph)
    - has no independent set of size 9
    
    We use 8 disjoint copies of K4: n=32, alpha=8, trivially C5-free.
    This is the best known lower bound for R(C5, K9).
    """
    n = 32
    edges = []
    for b in range(8):
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([4 * b + i, 4 * b + j])
    return {'n': n, 'edges': edges}
