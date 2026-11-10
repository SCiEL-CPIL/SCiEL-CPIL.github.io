import time

def construct():
    """
    Constructs a simple undirected graph on vertices 0..14 that maximizes edge count
    while avoiding equal-degree endpoints for any simple 3-edge path.
    
    Strategy:
    Use a complete bipartite graph K_{7,8}. 
    In K_{a,b}, any path of length 3 s-u-v-t has endpoints s and t in different partitions.
    Degrees in partition A are all b.
    Degrees in partition B are all a.
    If a != b, then deg(s) != deg(t) for all such paths.
    For n=15, max(a*b) with a+b=15 and a!=b is 7*8=56.
    """
    n = 15
    # Partition vertices into two sets of size 7 and 8
    # Set A: 0..6 (size 7)
    # Set B: 7..14 (size 8)
    
    edges = []
    
    # Add all edges between Set A and Set B
    for u in range(7):
        for v in range(7, 15):
            edges.append([u, v])
            
    # The result must be a list of lists of integers
    # Ensure types are Python int
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
        
    return result
