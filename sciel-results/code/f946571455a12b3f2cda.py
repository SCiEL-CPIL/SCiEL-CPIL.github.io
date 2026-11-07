"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

# Build adjacency and edge set for fast lookup
_edges = P['edges']
_n = P['n']
edge_set = set()
adj = defaultdict(set)
for u, v in _edges:
    edge_set.add((u, v))
    edge_set.add((v, u))
    adj[u].add(v)
    adj[v].add(u)

def _greedy_biclique(remaining_edges, adj):
    """Find a large biclique from remaining_edges using greedy approach."""
    if not remaining_edges:
        return None, None
    
    # Pick the edge with the most common neighbors
    best_u, best_v = None, None
    best_score = -1
    for (u, v) in list(remaining_edges)[:50]:  # Sample for speed
        common = adj[u] & adj[v]
        score = len(common)
        if score > best_score:
            best_score = score
            best_u, best_v = u, v
    
    if best_u is None:
        return None, None
    
    # Build initial biclique
    L = {best_u}
    R = {best_v}
    
    # Expand L: vertices connected to all of R
    changed = True
    while changed:
        changed = False
        new_L = set(L)
        for v in R:
            new_L |= adj[v]
        # Keep only those in new_L that are connected to ALL of R
        L = {v for v in new_L if all(v2 in adj[v] for v2 in R)}
        if L != set(L) or len(L) > len(new_L):
            pass
        
        # Expand R: vertices connected to all of L
        new_R = set(R)
        for u in L:
            new_R |= adj[u]
        R = {v for v in new_R if all(u2 in adj[v] for u2 in L)}
    
    # Filter to only edges in remaining
    L = {v for v in L if any((v, w) in remaining_edges or (w, v) in remaining_edges for w in R)}
    R = {v for v in R if any((u, v) in remaining_edges or (v, u) in remaining_edges for u in L)}
    
    if not L or not R:
        return None, None
    
    return sorted(L), sorted(R)

def _validate_partition(bicliques, original_edges):
    """Check if bicliques form a valid partition of original_edges."""
    seen = set()
    for L, R in bicliques:
        if not L or not R:
            return False
        for u in L:
            for v in R:
                key = (min(u,v), max(u,v))
                if key not in edge_set:
                    return False
                if key in seen:
                    return False
                seen.add(key)
    return seen == {(min(u,v), max(u,v)) for u,v in original_edges}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit, 10s buffer
    
    # Initial: star decomposition as fallback
    fallback = [[[u], [v for a,v in _edges if a==u]] for u in range(_n) if any(a==u for a,v in _edges)]
    
    best_result = None
    best_count = len(fallback)
    
    # Greedy biclique extraction
    remaining = set((min(u,v), max(u,v)) for u,v in _edges)
    bicliques = []
    
    while remaining and time.time() < deadline:
        # Find a biclique in remaining edges
        rem_edges_list = [(u,v) for (u,v) in remaining if u < v]
        
        if not rem_edges_list:
            break
        
        # Greedy: pick edge, expand
        best_biclique_size = 0
        best_L, best_R = None, None
        
        # Try several candidate edges
        candidates = random.sample(rem_edges_list, min(len(rem_edges_list), 20))
        
        for (u, v) in candidates:
            # Build biclique starting from (u,v)
            L = {u}
            R = {v}
            
            for _ in range(10):  # Limited iterations
                # Expand L
                new_L = set(L)
                for rv in R:
                    new_L |= adj[rv]
                L = {x for x in new_L if all(y in adj[x] for y in R) and 
                     any((min(x,y), max(x,y)) in remaining for y in R)}
                
                # Expand R
                new_R = set(R)
                for lu in L:
                    new_R |= adj[lu]
                R = {x for x in new_R if all(y in adj[x] for y in L) and
                     any((min(y,x), max(y,x)) in remaining for y in L)}
            
            if not L or not R:
                continue
                
            size = len(L) * len(R)
            # Verify all cross edges are in remaining
            valid = True
            for lu in L:
                for rv in R:
                    key = (min(lu,rv), max(lu,rv))
                    if key not in remaining:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and size > best_biclique_size:
                best_biclique_size = size
                best_L, best_R = L, R
        
        if best_L is None:
            # Can't find a biclique, break remaining into individual edges
            for (u, v) in list(remaining):
                bicliques.append([sorted([u]), sorted([v])])
                remaining.discard((u,v))
            break
        
        bicliques.append([sorted(best_L), sorted(best_R)])
        for lu in best_L:
            for rv in best_R:
                remaining.discard((min(lu,rv), max(lu,rv)))
    
    # Handle any remaining edges as individual bicliques
    for (u, v) in remaining:
        bicliques.append([sorted([u]), sorted([v])])
    
    if len(bicliques) < best_count:
        best_result = bicliques
        best_count = len(bicliques)
    
    # Try to merge bicliques to reduce count
    if time.time() < deadline - 10:
        merged = True
        current = [list(bc) for bc in best_result]
        
        while merged and time.time() < deadline - 10:
            merged = False
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    L1, R1 = current[i]
                    L2, R2 = current[j]
                    
                    # Try merging: L = L1 ∪ L2, R = R1 ∩ R2 or similar
                    # Check if (L1∪L2) × (R1∩R2) is valid
                    R_inter = set(R1) & set(R2)
                    if R_inter:
                        L_union = list(set(L1) | set(L2))
                        R_new = list(R_inter)
                        # Check all cross edges are in edge_set
                        valid = all((min(u,v), max(u,v)) in edge_set for u in L_union for v in R_new)
                        if valid:
                            new_biclique = [L_union, R_new]
                            # Check no overlap with other bicliques
                            overlap = False
                            for k in range(len(current)):
                                if k == i or k == j:
                                    continue
                                Lk, Rk = current[k]
                                for u in L_union:
                                    for v in R_new:
                                        if (u in Lk and v in Rk) or (u in Rk and v in Lk):
                                            overlap = True
                                            break
                                    if overlap:
                                        break
                                if overlap:
                                    break
                            if not overlap:
                                current[i] = new_biclique
                                current.pop(j)
                                merged = True
                                break
                if merged:
                    break
        
        if len(current) < best_count:
            best_result = current
            best_count = len(current)
    
    # Final validation and conversion
    if best_result is not None:
        # Ensure all types are correct
        final = []
        for bc in best_result:
            L = [int(x) for x in bc[0]]
            R = [int(x) for x in bc[1]]
            if L and R:
                final.append([L, R])
        
        # Validate
        if _validate_partition(final, _edges):
            return final
    
    # Fallback to star decomposition
    return fallback
# EVOLVE-BLOCK-END
