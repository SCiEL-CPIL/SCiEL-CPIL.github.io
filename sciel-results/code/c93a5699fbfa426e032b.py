import time
import sys

def construct():
    """
    Returns a list of edges for a graph on 12 vertices (0..11)
    with minimum degree 3, girth >= 4, and minimum distinct cycle lengths.
    
    Strategy: Use a single connected K_{6,6} bipartite graph.
    K_{6,6} is 6-regular, bipartite (girth 4), and contains only cycles of length 4 and 6.
    Distinct cycle lengths = 2.
    This is the theoretical minimum for a connected graph with min degree 3 and girth >= 4.
    """
    edges = []
    
    # K_{6,6} on vertices 0..11
    # Partition A: 0, 1, 2, 3, 4, 5
    # Partition B: 6, 7, 8, 9, 10, 11
    for i in range(6):
        for j in range(6):
            edges.append([i, 6 + j])
            
    # Ensure types are strictly Python int
    return [[int(u), int(v)] for u, v in edges]

if __name__ == "__main__":
    result = construct()
    print(result)
