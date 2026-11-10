import time
import itertools

def construct():
    """
    Constructs a graph on 12 vertices (0..11) consisting of two disjoint K_{3,3} components.
    - Component 1: Vertices 0..5. Partitions {0,1,2} and {3,4,5}.
    - Component 2: Vertices 6..11. Partitions {6,7,8} and {9,10,11}.
    
    Properties:
    - Simple undirected graph.
    - Minimum degree 3 (each vertex connected to 3 others).
    - Girth 4 (bipartite, smallest cycle is 4).
    - Contains cycles.
    - Distinct simple cycle lengths: 4 and 6. (Count = 2).
    - This is believed to be optimal for minimizing distinct cycle lengths under these constraints.
    """
    edges = []
    
    # Component 1: K_{3,3} on vertices 0-5
    # Partitions A1 = {0, 1, 2}, B1 = {3, 4, 5}
    A1 = [0, 1, 2]
    B1 = [3, 4, 5]
    for u in A1:
        for v in B1:
            edges.append([int(u), int(v)])
            
    # Component 2: K_{3,3} on vertices 6-11
    # Partitions A2 = {6, 7, 8}, B2 = {9, 10, 11}
    A2 = [6, 7, 8]
    B2 = [9, 10, 11]
    for u in A2:
        for v in B2:
            edges.append([int(u), int(v)])
            
    # Ensure no duplicates (K3,3 construction doesn't create them, but good practice)
    unique_edges = []
    seen = set()
    for e in edges:
        key = tuple(sorted(e))
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)
            
    return unique_edges
