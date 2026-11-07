import time
import random

P = {'n': 16}

def construct():
    """
    Constructs the complete bipartite graph K_{8,8} on vertices 0..15.
    
    Reasoning for optimality:
    1. The graph must have diameter exactly 2.
    2. Every edge must be critical: removing it must increase diameter > 2 or disconnect.
    3. If a graph contains a triangle (u, v, w), the edge (u, v) has an alternative path u-w-v of length 2.
       Removing (u, v) leaves the distance between u and v as 2.
       Since the graph was diameter 2, and all other distances are <= 2, the new diameter is 2.
       Thus, the diameter did NOT increase above 2.
       Therefore, any edge in a triangle is not critical.
    4. Consequently, the graph must be triangle-free.
    5. By Mantel's Theorem, the maximum number of edges in a triangle-free graph on n vertices is floor(n^2 / 4).
    6. For n = 16, max edges = 16^2 / 4 = 64.
    7. The complete bipartite graph K_{8,8} is the unique graph achieving this bound.
    8. In K_{8,8}, removing any edge (u, v) with u in A and v in B:
       - The graph remains connected.
       - The distance between u and v becomes 3 (u -> a' -> b' -> v is not right, u -> b' -> a' -> v? No.
         u is in A. Neighbors are in B.
         v is in B. Neighbors are in A.
         Path u to v: u -> b' (in B) ... wait.
         u is connected to all B. v is connected to all A.
         If we remove (u,v), u is still connected to B \ {v}.
         v is still connected to A \ {u}.
         Is there a path of length 2?
         u -> x -> v.
         x must be neighbor of u (so x in B) and neighbor of v (so x in A).
         Intersection of A and B is empty.
         So no path of length 2.
         Distance is 3 (u -> b' -> a' -> v).
         So diameter becomes 3.
         Condition satisfied.
    
    Returns edges as a list of [u, v] pairs.
    """
    n = P['n']
    mid = n // 2
    set_a = list(range(mid))
    set_b = list(range(mid, n))
    
    edges = []
    for u in set_a:
        for v in set_b:
            # Ensure types are Python int
            edges.append([int(u), int(v)])
            
    return edges
