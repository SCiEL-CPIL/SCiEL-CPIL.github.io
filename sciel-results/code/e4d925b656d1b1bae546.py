"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    all_edges = list(edge_set)
    
    # Generate candidate bicliques
    # 1. Star bicliques: {v} x N(v)
    candidates = []
    for v in range(n):
        if adj[v]:
            candidates.append(([v], list(adj[v])))
    
    # 2. Try to find larger bicliques by checking pairs of vertices
    # For each pair (u,v), if u and v are not adjacent, check if they share neighbors
    # Actually, for a biclique [L,R], all pairs in LxR must be edges
    # Try: for each small set L, find common neighbors
    
    # Try bicliques of form [L, R] where L has 2-3 vertices
    for size in range(2, 4):
        for L in combinations(range(n), size):
            # Find common neighbors of all vertices in L
            common = set(adj[L[0]])
            for u in L[1:]:
                common &= adj[u]
            # Remove vertices in L from common (they can't be in R if in L)
            common -= set(L)
            if common:
                # Check that all cross pairs are edges (they are by construction of common neighbors)
                # But we need to make sure no vertex in L is in common
                R = list(common)
                candidates.append((list(L), R))
    
    # 3. Also try larger R sides for single-vertex L
    # Already covered by star bicliques
    
    # Remove duplicates (normalize)
    seen = set()
    unique_candidates = []
    for L, R in candidates:
        key = (frozenset(L), frozenset(R))
        if key not in seen:
            seen.add(key)
            unique_candidates.append((L, R))
    
    # Greedy set cover
    uncovered = set(all_edges)
    selected = []
    
    # Precompute which edges each candidate covers
    def edges_covered(L, R):
        cov = set()
        for u in L:
            for v in R:
                e = (min(u,v), max(u,v))
                if e in edge_set:
                    cov.add(e)
        return cov
    
    # Compute coverage for each candidate
    cand_coverage = []
    for L, R in unique_candidates:
        cov = edges_covered(L, R)
        if cov:
            cand_coverage.append((L, R, cov))
    
    # Greedy: pick candidate covering most uncovered edges
    deadline = time.time() + 75  # safety margin
    
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = 0
        for i, (L, R, cov) in enumerate(cand_coverage):
            count = len(cov & uncovered)
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1:
            break
        
        L, R, cov = cand_coverage[best_idx]
        selected.append((L, R))
        uncovered -= cov
    
    # If any edges remain uncovered (shouldn't happen with star bicliques), add them individually
    for e in uncovered:
        u, v = e
        selected.append(([u], [v]))
    
    # Convert to proper format
    result = []
    for L, R in selected:
        L_list = [int(x) for x in L]
        R_list = [int(x) for x in R]
        result.append([L_list, R_list])
    
    return result
# EVOLVE-BLOCK-END

