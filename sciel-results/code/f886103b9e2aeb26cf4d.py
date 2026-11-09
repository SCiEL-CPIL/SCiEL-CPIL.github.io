"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin under 150s kill
    
    n = 20
    num_colors = 20
    
    # Strategy: Construct a C4-free graph.
    # A C4-free graph has no 4-cycles.
    # The condition "every 4-cycle is rainbow" is vacuously true for C4-free graphs.
    # We want to maximize the number of edges in a C4-free graph on 20 vertices.
    # This is equivalent to finding a dense set of points in a geometry such that
    # no two points share more than one "line" (in the generalized sense), or simply
    # using a known C4-free graph construction like the polarity graph or random greedy.
    
    # Greedy approach:
    # 1. Start with empty graph.
    # 2. Iterate over all possible edges (u, v) in a randomized order.
    # 3. Check if adding edge (u, v) creates a 4-cycle.
    #    A 4-cycle is formed if there exists a vertex w such that (u, w) and (v, w) are in the graph?
    #    No, a 4-cycle is u-v-w-x-u.
    #    Adding edge (u, v) creates a 4-cycle if there exist vertices w, x such that
    #    (u, w), (w, x), (x, v) are already in the graph.
    #    More efficiently: 
    #    Let N(u) be neighbors of u, N(v) be neighbors of v.
    #    Adding (u, v) creates a C4 if there is a path of length 3 between u and v.
    #    i.e., exists w, x such that (u,w) in G, (w,x) in G, (x,v) in G.
    #    This is equivalent to: intersection of N(u) and N(v) is not sufficient (that's C3).
    #    We need to check if there is a common neighbor of u and v? No.
    #    If u and v share a neighbor w, that's a triangle (C3), not C4.
    #    C4: u-w-x-v-u.
    #    So we need w in N(u) and x in N(v) such that (w,x) is an edge.
    #    Also x in N(u) and w in N(v) such that (w,x) is an edge.
    #    Actually, simply: Does there exist a path u-w-x-v of length 3?
    #    Let's precompute for each pair (u,v) the number of length-3 paths between them.
    #    Let A be the adjacency matrix.
    #    Number of length-3 paths from u to v is (A^3)[u][v].
    #    If (A^3)[u][v] > 0, then adding edge (u,v) creates at least one C4.
    #    Wait, A^3 counts walks. A walk u-w-x-v of length 3 corresponds to a C4 only if u!=x and w!=v and u!=v and w!=x.
    #    Since graph is simple, u!=v.
    #    If w=u, then u-u-x-v (not a simple path).
    #    If x=v, then u-w-v-v (not simple).
    #    If w=x, then u-w-w-v (not simple).
    #    So any walk of length 3 in a simple graph without loops corresponds to a C4 (possibly with repeated vertices? No, simple graph).
    #    Actually, if w=v, u-v-v-v? No, simple graph has no multi-edges.
    #    Let's just check explicitly for small n.
    
    # Efficient check for adding edge (u, v):
    # For each w in N(u):
    #    For each x in N(v):
    #       if (w, x) is an edge:
    #           return False (creates C4 u-w-x-v-u)
    # Note: We must ensure u, w, x, v are distinct.
    # w != u (since w in N(u))
    # x != v (since x in N(v))
    # w != v (if w==v, then v in N(u), so (u,v) already exists? No, we are adding (u,v). If v in N(u), (u,v) exists. So we only consider non-edges. Thus w!=v is guaranteed if (u,v) not in G? No. N(u) contains neighbors. If (u,v) not in G, v not in N(u). So w!=v is guaranteed.)
    # x != u (similarly, u not in N(v) if (u,v) not in G).
    # w != x? If w==x, then (u,w) and (w,v) are edges. This is a path u-w-v of length 2. Adding (u,v) creates a triangle. Triangles are allowed. The constraint is only on 4-cycles.
    # So we only care if w != x.
    
    # Algorithm:
    # 1. Initialize empty graph.
    # 2. Create list of all potential edges.
    # 3. Shuffle edges.
    # 4. For each edge (u,v) in shuffled order:
    #    Check if adding (u,v) creates a C4.
    #    If not, add it.
    # 5. Local search to improve? Or just greedy is good enough for C4-free max edges?
    #    Greedy C4-free construction on 20 vertices usually yields ~100-120 edges.
    #    Theoretical max for C4-free is O(n^(3/2)) ~ 20^1.5 ~ 89. But constants matter.
    #    Actually, Reiman's bound for C4-free is n/4 * (1 + sqrt(8n-7)) ~ 20/4 * (1+sqrt(153)) ~ 5 * 13.7 ~ 68.
    #    Wait, let's check known values.
    #    For n=20, max edges in C4-free graph is likely around 100-110?
    #    Let's check the bound again. Ex(n, C4) ~ n^(3/2)/2.
    #    20^1.5 / 2 = 89.4 / 2 = 44? No, 20^1.5 = 89.4. So ~45 edges?
    #    Let's look at specific constructions.
    #    Polarity graph of projective plane of order q has ~ (q+1)^2/2 vertices and ~ (q+1)^3/2 edges?
    #    For q=4, n=25, edges ~ 130.
    #    For n=20, we can take a subgraph of such a construction or use random greedy.
    #    Random greedy usually achieves close to the theoretical maximum for such extremal problems.
    
    # Let's implement the greedy check efficiently.
    adj = [set() for _ in range(n)]
    edges = []
    
    # Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(all_edges)
    
    def creates_c4(u, v):
        # Check if adding (u,v) creates a C4.
        # We need to find w, x such that (u,w) in G, (w,x) in G, (x,v) in G.
        # And w != x.
        # Iterate over neighbors of u
        for w in adj[u]:
            if w == v: continue # should not happen if (u,v) not in G
            # For each neighbor x of v
            for x in adj[v]:
                if x == u: continue
                if x == w: continue # triangle, allowed
                if w in adj[x]:
                    return True
        return False

    for u, v in all_edges:
        if time.time() > DEADLINE - 10:
            break
        if not creates_c4(u, v):
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v, 0]) # Color 0 is fine, no C4s exist
    
    # Now we have a C4-free graph.
    # We can try to add more edges by removing some?
    # Or just use this.
    # To maximize edges, we can try a local search:
    # Try to add an edge (u,v) that was rejected.
    # It was rejected because it created C4s.
    # If we remove one edge from each conflicting C4, we might be able to add (u,v) and net gain?
    # This is complex.
    
    # Alternative: Use a known dense C4-free graph construction.
    # The "Brown graph" or similar.
    # But greedy is often very good.
    
    # Let's verify the count.
    num_edges = len(edges)
    
    # If the greedy result is sparse, we can try to randomize the order multiple times.
    best_edges = edges
    best_count = num_edges
    
    for _ in range(5): # 5 restarts
        if time.time() > DEADLINE - 20:
            break
        adj = [set() for _ in range(n)]
        edges = []
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > DEADLINE - 10:
                break
            if not creates_c4(u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v, 0])
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    # Final check: ensure no 4-cycles.
    # Since we constructed it to be C4-free, it should be valid.
    # But let's double check with a simple loop just in case of bugs.
    # Actually, the logic `creates_c4` is correct for simple graphs.
    
    # Assign colors.
    # Since there are no 4-cycles, any coloring is valid.
    # We can just use color 0 for all edges, or distinct colors, doesn't matter.
    # The task says "edge-coloured with 20 colours".
    # "Every 4-cycle ... must receive four distinct colours".
    # If there are no 4-cycles, the condition is vacuously true.
    # So we can return the edges with any colors.
    # Let's use color 0 for all.
    
    return best_edges
# EVOLVE-BLOCK-END
