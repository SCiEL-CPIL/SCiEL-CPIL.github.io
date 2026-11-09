import time
import sys
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build original adjacency sets for validity checks
    orig_adj = [set() for _ in range(n)]
    for u, v in edges:
        orig_adj[u].add(v)
        orig_adj[v].add(u)
        
    # Track uncovered edges using sets of neighbors
    uncovered = [set() for _ in range(n)]
    for u, v in edges:
        uncovered[u].add(v)
        uncovered[v].add(u)
        
    bicliques = []
    
    def find_best_biclique():
        """
        Find the biclique that covers the most uncovered edges.
        We search over small left sets L and take R as the common uncovered neighbors.
        """
        # Get vertices with uncovered edges
        active = [u for u in range(n) if uncovered[u]]
        if not active:
            return None, None, 0
            
        # Sort by degree to prioritize high-degree vertices for L
        active_sorted = sorted(active, key=lambda x: len(uncovered[x]), reverse=True)
        
        best_L = None
        best_R = None
        max_covered = 0
        
        # We'll try L sizes 1, 2, and maybe 3 if fast enough
        # For L size 1:
        # R = uncovered[u]
        # Covered = len(R)
        
        # For L size 2: {u, v}
        # R = uncovered[u] & uncovered[v] - {u, v}
        # Covered = 2 * len(R)
        
        # For L size 3: {u, v, w}
        # R = uncovered[u] & uncovered[v] & uncovered[w] - {u, v, w}
        # Covered = 3 * len(R)
        
        # We limit the search space to keep it fast.
        # For L size 1, check all active.
        for u in active_sorted:
            R = set(uncovered[u])
            covered = len(R)
            if covered > max_covered:
                max_covered = covered
                best_L = [u]
                best_R = list(R)
                
        # For L size 2, check pairs among top K active vertices
        # K should be small, e.g., 10-20
        K = 15
        top_active = active_sorted[:K]
        
        for i in range(len(top_active)):
            u = top_active[i]
            for j in range(i+1, len(top_active)):
                v = top_active[j]
                # R must be common neighbors in uncovered graph
                R = uncovered[u].intersection(uncovered[v])
                R.discard(u)
                R.discard(v)
                if not R:
                    continue
                covered = 2 * len(R)
                if covered > max_covered:
                    max_covered = covered
                    best_L = [u, v]
                    best_R = list(R)
                    
        # For L size 3, check triples among top M active vertices
        # M should be even smaller, e.g., 5-8
        M = 8
        top_active_m = active_sorted[:M]
        
        # Time check
        if time.time() > deadline - 10:
             return best_L, best_R, max_covered
             
        for i in range(len(top_active_m)):
            u = top_active_m[i]
            for j in range(i+1, len(top_active_m)):
                v = top_active_m[j]
                common_uv = uncovered[u].intersection(uncovered[v])
                common_uv.discard(u)
                common_uv.discard(v)
                # w must be in common_uv to form a valid biclique with u, v
                # Actually, w just needs to have all of R as neighbors.
                # R = common_uv & uncovered[w] - {u, v, w}
                # To maximize R, w should be chosen from common_uv?
                # If w is in common_uv, then w is neighbor of u and v.
                # Then R = {x | x in common_uv and x in uncovered[w]}
                # Note: x cannot be u, v, w.
                
                # Let's iterate w from common_uv (limited)
                # If common_uv is large, limit it.
                w_candidates = list(common_uv)
                # Sort w_candidates by degree to pick good ones?
                # Or just take top few
                if len(w_candidates) > 10:
                    w_candidates = sorted(w_candidates, key=lambda x: len(uncovered[x]), reverse=True)[:10]
                    
                for w in w_candidates:
                    R = common_uv.intersection(uncovered[w])
                    R.discard(u)
                    R.discard(v)
                    R.discard(w)
                    if not R:
                        continue
                    covered = 3 * len(R)
                    if covered > max_covered:
                        max_covered = covered
                        best_L = [u, v, w]
                        best_R = list(R)
                        
        return best_L, best_R, max_covered

    while any(uncovered[u] for u in range(n)):
        if time.time() > deadline:
            break
            
        best_L, best_R, max_covered = find_best_biclique()
        
        if best_L is None:
            # Fallback: pick any edge
            for u in range(n):
                if uncovered[u]:
                    v = next(iter(uncovered[u]))
                    best_L = [u]
                    best_R = [v]
                    break
            if best_L is None:
                break
        
        # Validate biclique against original graph (just in case, though logic ensures it)
        # Every pair (u in L, v in R) must be an edge in original graph
        valid = True
        for u in best_L:
            for v in best_R:
                if u != v and v not in orig_adj[u]:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # This shouldn't happen if we derived R from uncovered sets which are subsets of orig_adj
            # But if it does, fallback to single edge
            for u in range(n):
                if uncovered[u]:
                    v = next(iter(uncovered[u]))
                    best_L = [u]
                    best_R = [v]
                    break
        
        bicliques.append([best_L, best_R])
        
        # Remove covered edges from uncovered
        for u in best_L:
            for v in best_R:
                if v in uncovered[u]:
                    uncovered[u].discard(v)
                if u in uncovered[v]:
                    uncovered[v].discard(u)
                    
    # Convert to required format
    result = []
    for L, R in bicliques:
        result.append([[int(x) for x in L], [int(y) for y in R]])
        
    return result
