import time
import itertools

P = {'n': 16}

def construct():
    """
    Return a maximum-edge simple undirected graph on vertices 0..15 with:
      - diameter exactly 2
      - every edge essential (removing any single edge makes diameter > 2 or disconnects)
    
    The optimal construction is the complete bipartite graph K_{8,8}, which has 64 edges.
    
    Proof sketch:
      - K_{8,8} is triangle-free, so for any edge (u,v), there is no common neighbor.
        Removing (u,v) makes d(u,v) = 3 (via u-w-x-v where w is in u's part, x in v's part).
        Thus diameter becomes 3 > 2, so every edge is essential.
      - Any graph with more than 64 edges on 16 vertices contains a triangle (Mantel's theorem).
        In any triangle (a,b,c), the edge (a,b) is not essential: removing it leaves path a-c-b,
        so d(a,b) remains 2, and all other distances are unchanged or improved.
        Therefore, no graph with > 64 edges can satisfy the edge-essentiality condition.
    
    We construct K_{8,8} with parts {0..7} and {8..15}.
    """
    n = P['n']
    deadline = time.time() + 85  # safety margin before 90s kill
    
    mid = n // 2
    edges = []
    for u in range(mid):
        for v in range(mid, n):
            edges.append([u, v])
    
    # Return as list of lists of Python ints
    return edges
