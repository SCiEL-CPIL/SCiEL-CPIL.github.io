"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return a dense graph on 16 vertices (0..15) with diameter exactly 2,
    such that deleting ANY individual edge either disconnects the graph
    or increases the diameter above 2. Maximize edge count.
    
    Key insight: We need a graph where every edge is "essential" for maintaining
    diameter 2. This means for every edge (u,v), there is NO alternative path
    of length 2 between u and v (otherwise removing (u,v) would still keep
    dist(u,v) <= 2).
    
    If there's a path u-w-v of length 2, then removing edge (u,v) doesn't increase
    dist(u,v) beyond 2. So we need: for every edge (u,v), there is NO common neighbor
    of u and v.
    
    This means the graph must be triangle-free (no two adjacent vertices share a common
    neighbor). But wait - if u and v are adjacent, and they share a common neighbor w,
    then we have triangle u-v-w-u. So the graph must be triangle-free.
    
    But we also need diameter exactly 2. In a triangle-free graph, for any non-adjacent
    pair (u,v), there must be exactly one common neighbor (to maintain diameter 2 without
    creating triangles that would make edges non-essential).
    
    Actually, let me reconsider. The condition is:
    - For every edge (u,v): removing it must increase diameter > 2 or disconnect.
      This happens iff there is no path of length 2 from u to v (i.e., u and v have no
      common neighbor).
    
    So the graph must be triangle-free (since if u~v and u~w and v~w, then w is a common
    neighbor of u and v, making edge (u,v) non-essential).
    
    Additionally, for diameter to be exactly 2:
    - For every non-adjacent pair (u,v), there must be at least one common neighbor.
    
    So we need a triangle-free graph on 16 vertices with diameter 2, maximizing edges.
    
    The complete bipartite graph K_{8,8} is triangle-free and has diameter 2.
    It has 8*8 = 64 edges.
    
    Can we do better? Let's think...
    
    In a triangle-free graph, by Mantel's theorem, the maximum number of edges is
    floor(n^2/4) = floor(256/4) = 64. So K_{8,8} achieves the maximum possible
    number of edges for a triangle-free graph on 16 vertices.
    
    Since our graph MUST be triangle-free (for all edges to be essential), 64 is the
    maximum possible edge count.
    
    Let's verify K_{8,8} works:
    - Vertices: A = {0,...,7}, B = {8,...,15}
    - Edges: all pairs (a,b) where a in A, b in B
    - Triangle-free: yes, bipartite graphs are triangle-free
    - Diameter 2: any two vertices in the same part have distance 2 (via any vertex
      in the other part). Adjacent vertices have distance 1. So diameter is 2.
    - Every edge essential: for edge (a,b) with a in A, b in B, is there a common
      neighbor? A common neighbor of a and b would need to be adjacent to both.
      a is only adjacent to vertices in B, b is only adjacent to vertices in A.
      So a common neighbor would need to be in A∩B = ∅. No common neighbor exists.
      Thus removing (a,b) makes dist(a,b) infinite (disconnected), so diameter > 2.
    
    This is optimal with 64 edges.
    """
    n = 16
    mid = n // 2  # 8
    
    edges = []
    for u in range(mid):
        for v in range(mid, n):
            edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
