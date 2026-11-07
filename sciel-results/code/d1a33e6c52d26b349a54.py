def construct():
    """
    Constructs the complete bipartite graph K_{8,8} on 16 vertices.
    
    Reasoning:
    1. The task requires a graph with diameter exactly 2 where every edge is critical
       (deleting any single edge increases diameter > 2 or disconnects the graph).
    2. If an edge (u, v) is part of a triangle (i.e., there exists w adjacent to both u and v),
       then removing (u, v) still leaves a path u-w-v of length 2. Thus, the distance d(u,v)
       remains 2. If all other pairs remain within distance 2 (which is likely in a dense graph),
       the diameter might not increase. To guarantee that removing ANY edge increases the diameter
       or disconnects, the graph must be triangle-free.
    3. By Mantel's Theorem, the maximum number of edges in a triangle-free graph on n vertices
       is floor(n^2 / 4).
    4. For n=16, the maximum is 16^2 / 4 = 64 edges, achieved uniquely by K_{8,8}.
    5. In K_{8,8}, the vertices are partitioned into two sets A and B of size 8.
       Edges only exist between A and B.
       - Diameter is 2: Any two nodes in the same partition are distance 2 apart (via any node in the other partition). Any two nodes in different partitions are distance 1 apart.
       - Criticality: Consider an edge (u, v) with u in A, v in B.
         N(u) = B, N(v) = A.
         N(u) ∩ N(v) = ∅.
         Removing (u, v) eliminates the direct path.
         Is there a path of length 2? A path u-x-v requires x adjacent to u and v.
         Since N(u) ∩ N(v) = ∅, no such x exists.
         The shortest path from u to v becomes length 3: u -> a' (in A, a'≠u) -> b' (in B, b'≠v) -> v.
         Wait, u is in A. Neighbors of u are in B. Let w be a neighbor of u (w in B, w≠v).
         Neighbors of w are in A. Let z be a neighbor of w (z in A, z≠u).
         Neighbors of z are in B. v is in B, so z is adjacent to v.
         Path: u -> w -> z -> v. Length 3.
         Thus, d(u, v) becomes 3. The diameter of the graph becomes 3 (since other pairs are still dist 1 or 2).
         Therefore, the diameter increases from 2 to 3.
       - Hence, K_{8,8} satisfies all conditions and maximizes edge count.
    """
    n = 16
    half = n // 2
    part1 = list(range(half))
    part2 = list(range(half, n))
    
    edges = []
    for u in part1:
        for v in part2:
            # Ensure strict Python int types
            edges.append([int(u), int(v)])
            
    return edges
