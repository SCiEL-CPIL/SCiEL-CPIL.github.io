import time

def construct():
    n = 16
    # The maximum number of edges in a graph on n vertices with diameter 2 
    # such that every edge is essential (removing it increases diameter or disconnects)
    # is achieved by the complete bipartite graph K_{n/2, n/2} for even n.
    # This graph is triangle-free. In a triangle-free graph, no edge is part of a 
    # 2-path alternative, so removing an edge increases the distance between its 
    # endpoints from 1 to >1 (specifically 3 in K_{8,8}), increasing the diameter 
    # from 2 to 3.
    
    # Partition vertices into two sets of size n/2
    half = n // 2
    set_a = list(range(half))
    set_b = list(range(half, n))
    
    edges = []
    for u in set_a:
        for v in set_b:
            # Ensure u < v is not required, just unique pairs
            # u is from 0..7, v is from 8..15, so u != v always
            edges.append([u, v])
            
    return edges
