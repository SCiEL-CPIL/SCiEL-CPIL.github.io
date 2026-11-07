"""Rewritten program: construct a triangle-free graph on 12 vertices with min degree >= 3,
minimizing distinct simple cycle lengths. Uses a single connected 3-regular bipartite graph."""

def construct():
    # Construct two disjoint K3,3 components.
    # K3,3 is 3-regular, bipartite, and has only cycle lengths 4 and 6.
    # This minimizes the number of distinct cycle lengths to 2.
    edges = []
    
    # Component 1: Vertices 0, 1, 2 connected to 3, 4, 5
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.append([u, v])
            
    # Component 2: Vertices 6, 7, 8 connected to 9, 10, 11
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.append([u, v])
            
    return edges
