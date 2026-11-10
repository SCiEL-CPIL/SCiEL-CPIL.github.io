"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a dense graph on 16 vertices with diameter exactly 2, where every edge is critical.
    Strategy:
    1. Start with a complete bipartite graph K_{8,8}.
    2. This graph has diameter 2.
    3. Verify that every edge is critical (removing it disconnects the graph or increases diameter > 2).
       - In K_{8,8}, removing an edge (u, v) leaves u and v disconnected via direct edge.
       - Since it's bipartite and we removed the only edge between them, they are at distance 3 (u-neighbor-v_neighbor? No, u is in A, v in B. u connects to all B except v. v connects to all A except u. Path u -> u' (in A? No, u' in B) ... 
       Wait, u in A, v in B.
       Neighbors of u in A: none (bipartite). Neighbors of u in B: all except v.
       Neighbors of v in A: all except u. Neighbors of v in B: none.
       Shortest path from u to v after removing (u,v):
       u -> w (in B, w != v) -> z (in A, z != u) -> v.
       Length 3.
       So diameter becomes 3.
       Thus every edge is critical.
    4. The number of edges is 64.
    
    Is it possible to have more than 64 edges?
    If we add any edge within A (say (a1, a2)), we create a triangle (a1, b, a2) for any b in B.
    Consider edge (a1, b) in the original K_{8,8}.
    If we remove (a1, b), is the diameter still 2?
    a1 is connected to a2 (new edge). a2 is connected to b (original edge).
    So dist(a1, b) = 2 via a2.
    Thus, removing (a1, b) does NOT increase the distance to >2.
    So (a1, b) is NOT critical.
    Therefore, we cannot simply add edges to K_{8,8} and keep all edges critical.
    
    What if we start with a different graph?
    The condition "removing EACH individual edge must increase its diameter above two or disconnect it" is very strong.
    This implies that for every edge (u, v), dist(u, v) in G - (u, v) must be > 2 or infinite.
    This means that in G, u and v must NOT have a common neighbor.
    Proof: If u and v have a common neighbor w, then (u, w) and (w, v) are edges.
    If we remove (u, v), the path u-w-v still exists, so dist(u, v) <= 2.
    Unless the diameter of G - (u, v) becomes > 2 due to some OTHER pair?
    The requirement is: "Deleting EACH individual edge must increase its diameter above two OR disconnect it."
    Let G' = G - (u, v).
    Condition: diam(G') > 2 or G' is disconnected.
    
    If u and v have a common neighbor w, then dist_{G'}(u, v) <= 2.
    This does not necessarily mean diam(G') <= 2.
    However, if G is "diameter 2 critical", usually it implies that the edge is essential for the distance 2 property of its endpoints.
    But the text says "increase ITS diameter". "Its" refers to the graph.
    
    Let's check the definition carefully.
    "Deleting EACH individual edge must increase its diameter above two or disconnect it."
    
    Counter-example to "must be triangle-free":
    Consider a cycle C5. Diameter 2.
    Edges: (0,1), (1,2), (2,3), (3,4), (4,0).
    Remove (0,1). Graph is path 1-2-3-4-0. Diameter 4. > 2. Critical.
    C5 has 5 edges.
    
    Consider K4. Diameter 1. Not diameter 2.
    
    Consider the Petersen graph (10 vertices, 15 edges). Diameter 2.
    Is it edge critical?
    Take an edge (u, v). Do they have a common neighbor? No, girth 5.
    Remove (u, v). dist(u, v) becomes 3?
    In Petersen, diameter is 2.
    If we remove an edge, does diameter stay 2?
    Actually, Petersen is not edge critical in the sense that some edges might be redundant for diameter 2?
    No, if girth > 4, then no triangles or 4-cycles.
    If no common neighbor, removing (u, v) increases dist(u, v) to >= 3.
    Does it increase the GLOBAL diameter?
    If there is another pair at distance 2, and we remove (u, v), their distance might stay 2.
    But if dist(u, v) becomes 3, then the diameter of G' is at least 3.
    So if G has diameter 2, and G' has diameter >= 3, then diam(G') > 2.
    So ANY graph with diameter 2 and girth >= 5 is edge critical?
    Wait. If G has diameter 2, and we remove (u, v), and dist(u, v) becomes 3, then diam(G') >= 3.
    So diam(G') > 2.
    So the condition is satisfied.
    
    So, we want a graph with:
    1. n=16
    2. Diameter 2
    3. Girth >= 5 (to ensure no common neighbors for adjacent vertices)
       OR more generally: for every edge (u, v), they have no common neighbor.
       (Because if they have a common neighbor, removing (u, v) leaves dist(u, v) <= 2, but maybe other pairs go to 3?
        If dist(u, v) <= 2 in G', it doesn't mean diam(G') <= 2.
        But if G' is still diameter 2, then it fails.
        If G' has diameter 3, it passes.
        So having a common neighbor doesn't automatically disqualify an edge.
        It only disqualifies it if the rest of the graph remains "diameter 2 tight" enough.
        However, if they have a common neighbor, it's safer to assume the edge is NOT critical unless the graph is very sparse.
        Actually, if (u, v) has a common neighbor w, then (u, w, v) is a path of length 2.
        If G' still has diameter 2, then (u, v) is not critical.
        If G' has diameter 3, then (u, v) IS critical (because diameter increased from 2 to 3).
        
        So the condition "no common neighbor" is sufficient but not necessary.
        
        However, to MAXIMIZE edges, we want as many edges as possible.
        If we add an edge (u, v), and they already have a common neighbor, then (u, v) is likely non-critical.
        Why? Because removing (u, v) leaves the path u-w-v.
        Unless removing (u, v) breaks ALL length-2 paths for some OTHER pair?
        Unlikely in a dense graph.
        
        So, to maximize edges, we should probably stick to graphs where NO two adjacent vertices share a common neighbor (i.e., girth >= 5).
        This is a "diameter 2, girth 5" graph.
        Or, if we allow common neighbors, we need to be very careful.
        
        Let's check the maximum number of edges in a graph on 16 vertices with diameter 2 and girth 5.
        This is related to the Moore bound.
        For diameter d=2, girth g=5, max edges?
        
        Known graphs:
        - Hoffman-Singleton graph: 50 vertices, 240 edges.
        - Petersen: 10 vertices, 15 edges.
        - Clebsch graph: 16 vertices, 48 edges.
          - The Clebsch graph is a strongly regular graph srg(16, 5, 0, 2).
          - It has diameter 2.
          - It has girth 4? No, lambda=0 means no triangles. mu=2 means any two non-adjacent vertices have 2 common neighbors.
          - Since lambda=0, no triangles.
          - Does it have 4-cycles?
          - Yes, srg(16, 5, 0, 2) has 4-cycles.
          - If it has 4-cycles, then adjacent vertices do NOT share a common neighbor (girth 5? No, girth 4 means there IS a 4-cycle, which implies non-adjacent vertices share neighbors, but adjacent vertices?
          - In a triangle-free graph, adjacent vertices share a common neighbor iff there is a triangle? No.
          - u-v edge. u-w-v path. w is a common neighbor of u and v.
          - This forms a triangle u-w-v-u? No, u-v is an edge. So u, v, w form a triangle.
          - So if lambda=0 (no triangles), then adjacent vertices have NO common neighbors.
          - So the Clebsch graph has girth 5?
          - Wait, if it has 4-cycles, the girth is 4.
          - But 4-cycles involve non-adjacent vertices.
          - Adjacent vertices in a triangle-free graph have no common neighbors.
          - So for every edge (u, v), there is no w such that (u, w) and (v, w) are edges.
          - Therefore, removing (u, v) increases dist(u, v) to at least 3.
          - Since the graph was diameter 2, and dist(u, v) becomes >= 3, the new diameter is >= 3.
          - So the Clebsch graph satisfies the condition!
          
          Edge count of Clebsch graph: 48.
          K_8,8 edge count: 64.
          
          K_8,8 is triangle-free (lambda=0).
          Does K_8,8 have common neighbors for adjacent vertices?
          u in A, v in B.
          Common neighbors would be in A or B?
          Neighbor of u is in B. Neighbor of v is in A.
          Intersection is empty.
          So K_8,8 also has no common neighbors for adjacent vertices.
          
          So both K_8,8 and Clebsch are valid.
          K_8,8 has 64 edges.
          Clebsch has 48 edges.
          
          Can we do better than 64?
          Mantel's theorem says max edges in triangle-free graph is n^2/4 = 64.
          If we add ANY edge to K_8,8, we create a triangle.
          If we create a triangle (u, v, w), then consider edge (u, v).
          u and v have common neighbor w.
          If we remove (u, v), dist(u, v) becomes 2 (via w).
          Does the diameter increase?
          Only if some other pair's distance becomes > 2.
          But (u, v) still has distance 2.
          If the rest of the graph is "robust", the diameter stays 2.
          In K_8,8 + (a1, a2), remove (a1, b).
          a1 and b have common neighbor a2.
          dist(a1, b) = 2.
          Are any other pairs at distance > 2?
          Probably not, the graph is very dense.
          So (a1, b) is NOT critical.
          
          Thus, we are limited to triangle-free graphs if we want to be safe.
          And the max edges for triangle-free is 64.
          
          So 64 is the likely maximum.
    """
    n = 16
    partition_a = list(range(8))
    partition_b = list(range(8, 16))
    
    edges = []
    for u in partition_a:
        for v in partition_b:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
