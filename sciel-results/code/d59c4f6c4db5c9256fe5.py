import time

def construct():
    """
    Return a list of edges [u, v] for a graph on vertices 0..15 that:
    1. Has diameter exactly 2.
    2. Every edge is critical: removing any single edge increases the diameter above 2
       (or disconnects the graph).
    3. Maximizes the number of edges.
    
    The condition "removing any edge increases diameter above 2" is equivalent to:
    - The graph is triangle-free.
      Proof: 
        - If the graph contains a triangle (u, v, w), then removing edge (u, v) leaves a 
          path u-w-v of length 2. Thus, dist(u, v) in G-e is 2. Since the graph originally 
          had diameter 2, the new diameter is at most 2 (it can't be less than 2 if other 
          pairs are at distance 2, but specifically for the pair (u,v) it's 2). 
          Wait, the condition is "diameter increases ABOVE two". 
          If dist(u,v) becomes 2, the diameter of G-e might still be 2 (if some other pair 
          was at distance 2 and remains so, or if no pair becomes >2). 
          Actually, if G has diameter 2, and we remove an edge in a triangle, the distance 
          between the endpoints of that edge becomes 2. The diameter of G-e is the max 
          distance between any pair. Since all distances were <= 2 in G, and we only 
          increased the distance of one pair to 2 (it was 1), all other distances remain 
          <= 2. So the diameter of G-e is exactly 2 (assuming G was connected and not a 
          complete graph where diam=1, but diam=2 implies some pair at dist 2). 
          So if there is a triangle, removing an edge in that triangle does NOT increase 
          the diameter above 2. It stays 2. Thus, the edge is NOT critical.
        - Conversely, if the graph is triangle-free, removing edge (u, v) removes the only 
          path of length 1. Any path of length 2 between u and v would require a common 
          neighbor w such that (u,w) and (w,v) are edges. This forms a triangle u-w-v-u. 
          Since the graph is triangle-free, no such w exists. Thus, no path of length 1 or 2 
          exists in G-e. The distance must be at least 3 (if connected) or infinite (if 
          disconnected). In either case, the diameter increases above 2 (or becomes infinite).
    
    Therefore, the problem reduces to finding a triangle-free graph on 16 vertices with 
    diameter 2 that maximizes the number of edges.
    
    By Mantel's Theorem, the maximum number of edges in a triangle-free graph on n vertices 
    is floor(n^2 / 4). For n=16, this is 64.
    The unique extremal graph is the complete bipartite graph K_{8,8}.
    
    Let's verify K_{8,8} satisfies the conditions:
    1. Triangle-free? Yes, bipartite graphs are triangle-free.
    2. Diameter 2? Yes, any two vertices in the same partition have a common neighbor in 
       the other partition (since it's complete bipartite). Any two vertices in different 
       partitions are adjacent. So max distance is 2.
    3. Every edge critical? Yes, by the triangle-free argument above.
    
    Thus, K_{8,8} is the optimal solution with 64 edges.
    """
    
    n = 16
    half = n // 2
    edges = []
    
    for i in range(half):
        for j in range(half, n):
            edges.append([i, j])
            
    return edges
