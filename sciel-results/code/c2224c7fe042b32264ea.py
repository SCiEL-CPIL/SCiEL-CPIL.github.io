"""
Constructs a graph on 16 vertices (0..15) with diameter exactly 2,
where every edge is essential (removing any single edge increases
diameter above 2 or disconnects the graph).

The optimal graph is the complete bipartite graph K_{8,8}.
- It has 64 edges.
- It is triangle-free, so for every edge (u, v), N(u) and N(v) are disjoint.
- Removing any edge (u, v) makes the distance between u and v equal to 3
  (path u -> b' -> a' -> v where b' in B\{v}, a' in A\{u}).
- Thus, the diameter increases to 3.
- By Mantel's Theorem, 64 is the maximum number of edges in a triangle-free
  graph on 16 vertices. Any graph with more edges must contain a triangle,
  and edges in triangles are generally not essential for diameter 2 in dense graphs.
"""

def construct():
    n = 16
    half = n // 2
    edges = []
    
    # Construct K_{8,8}
    # Partitions: 0..7 and 8..15
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
            
    return edges
