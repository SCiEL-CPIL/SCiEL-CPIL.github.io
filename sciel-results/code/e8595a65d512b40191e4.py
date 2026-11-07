"""
Constructs a graph on 16 vertices with diameter 2 such that removing any edge increases the diameter.
The optimal structure is the complete bipartite graph K_{8,8}, which achieves the
Mantel bound of 64 edges for triangle-free graphs on 16 vertices.

In a triangle-free graph with diameter 2, every edge is critical:
- If we remove edge (u, v), since there is no triangle, there is no common neighbor w 
  such that u-w-v is a path of length 2. 
- Thus, the distance between u and v becomes > 2.
- In K_{8,8}, removing (u, v) where u in A, v in B:
  Neighbors of u are in B, neighbors of v are in A.
  Intersection of N(u) and N(v) is empty.
  So dist(u, v) > 2 in G - (u, v).
  Specifically, dist(u, v) = 3 (e.g., u -> b_other -> a_other -> v).
  Thus diameter becomes 3.
"""

def construct():
    n = 16
    edges = []
    
    # K_{8,8}: bipartition {0..7} and {8..15}
    # This graph is triangle-free.
    # In a triangle-free graph with diameter 2, every edge is critical:
    # If we remove edge (u, v), since there is no triangle, there is no common neighbor w 
    # such that u-w-v is a path of length 2. 
    # Thus, the distance between u and v becomes > 2 (specifically 3 or disconnected) 
    # provided the rest of the graph doesn't provide a shorter path. 
    # In K_{8,8}, removing (u, v) where u in A, v in B:
    # Any path from u to v must go u -> a' (in A? no, u is in A) -> b' (in B) -> ... 
    # Wait, neighbors of u are in B. Neighbors of v are in A.
    # A path u -> ... -> v of length 2 would require a common neighbor.
    # Neighbors of u are B \ {v} (if we consider the graph without the edge, but in the original graph N(u)=B).
    # Neighbors of v are A.
    # Intersection of N(u) and N(v) is empty because N(u) subset B and N(v) subset A, and A cap B = empty.
    # So dist(u, v) > 2 in G - (u, v).
    # Specifically, dist(u, v) = 3 (e.g., u -> b_other -> a_other -> v is not right. 
    # u is in A. N(u) = B. v is in B. N(v) = A.
    # Path: u -> b1 (in B) ... wait u is connected to all in B.
    # If we remove (u, v), u is still connected to all other b in B.
    # v is still connected to all other a in A.
    # To get from u to v: u -> b_other (in B) is not useful to reach v directly.
    # u -> b_other. b_other is connected to all a in A.
    # So u -> b_other -> a_other. a_other is connected to v.
    # So path u -> b_other -> a_other -> v has length 3.
    # Thus diameter becomes 3.
    
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
            
    return edges
