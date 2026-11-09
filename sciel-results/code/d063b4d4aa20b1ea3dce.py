import time
import random
from itertools import combinations

def _has_c5(adj, n):
    """Check if graph with adjacency matrix adj on n vertices has a C5 subgraph."""
    # Precompute cycles on 5 vertices
    # There are 12 distinct cycles on 5 labeled vertices
    cycles = [
        (0,1,2,3,4),
        (0,1,2,4,3),
        (0,1,3,2,4),
        (0,1,3,4,2),
        (0,1,4,2,3),
        (0,1,4,3,2),
        (0,2,1,3,4),
        (0,2,1,4,3),
        (0,2,3,1,4),
        (0,2,4,1,3),
        (0,3,1,2,4),
        (0,3,4,2,1),
    ]
    
    for combo in combinations(range(n), 5):
        for cyc in cycles:
            ok = True
            for i in range(5):
                u = combo[cyc[i]]
                v = combo[cyc[(i+1)%5]]
                if not adj[u][v]:
                    ok = False
                    break
            if ok:
                return True
    return False

def _has_independent_set_9(adj, n):
    """Check if there is an independent set of size 9."""
    for combo in combinations(range(n), 9):
        ok = True
        for i in range(9):
            for j in range(i+1, 9):
                u = combo[i]
                v = combo[j]
                if adj[u][v]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return True
    return False

def _is_valid(n, edge_list):
    adj = [[False]*n for _ in range(n)]
    for u, v in edge_list:
        adj[u][v] = True
        adj[v][u] = True
    if _has_c5(adj, n):
        return False
    if _has_independent_set_9(adj, n):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = 110.0
    
    # Best known valid graph: 8 disjoint K4s, n=32
    best_n = 32
    best_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    
    # Try to find n=33
    # Strategy: Add vertex 32 to 8 K4s.
    # Connect 32 to a subset S of vertices in the 32-vertex graph.
    # We want to avoid C5 and alpha >= 9.
    
    # Random search over connection patterns
    # To be fast, we only try connecting 32 to vertices in one K4 block or a few.
    # If we connect to multiple blocks, C5 risk increases if paths exist.
    # In disjoint K4s, no paths between blocks.
    # So if we connect 32 to vertices in K4_0 and K4_1, no C5 involving both unless there's a path between them (there isn't).
    # So C5 can only form within the neighborhood of 32 in a single component if that component has cycles?
    # K4 is complete.
    # If 32 connects to S in K4_b, C5 forms if S contains endpoints of a path of length 3 in K4_b.
    # In K4, any two distinct vertices are endpoints of a path of length 3 (via the other two).
    # So if |S| >= 2, C5 is formed.
    # Thus, 32 can be connected to at most 1 vertex in ANY K4_b to avoid C5.
    
    # If 32 connects to at most 1 vertex in each K4_b:
    # Let S_b be the set of neighbors of 32 in K4_b. |S_b| <= 1.
    # To keep alpha <= 8:
    # Independent set can include 32 and one vertex from each K4_b that is NOT in S_b.
    # If |S_b| = 0, we can pick 1 from K4_b.
    # If |S_b| = 1, we can pick 1 from K4_b \ S_b (size 3).
    # So we can always pick 1 from each K4_b.
    # Total size 1 (for 32) + 8 (one from each K4) = 9.
    # So this strategy always yields alpha >= 9.
    
    # Conclusion: We cannot extend 8 K4s by 1 vertex without creating C5 or alpha >= 9.
    # We need a different base or structure.
    
    # Given the time constraint and difficulty, we return the n=32 graph.
    
    return {'n': best_n, 'edges': best_edges}
