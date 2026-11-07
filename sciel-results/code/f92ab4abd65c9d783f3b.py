"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_host_adj(n, edges):
    host_adj = [set() for _ in range(n)]
    for u, v in edges:
        host_adj[u].add(v)
        host_adj[v].add(u)
    return host_adj

def _expand_biclique(start_v, remaining_edges, n, max_iter=100):
    """Expand a biclique starting from vertex start_v.
    Returns (L, R, removed_count) where L and R are sets of vertices."""
    # Find neighbors of start_v in remaining graph
    R = set()
    for e in remaining_edges:
        if start_v in e:
            other = list(e)[0] if list(e)[0] != start_v else list(e)[1]
            R.add(other)
    
    if not R:
        return set(), set(), 0
    
    L = {start_v}
    
    # Build adjacency in remaining graph for fast lookups
    rem_adj = [set() for _ in range(n)]
    for e in remaining_edges:
        u, v = list(e)
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    
    changed = True
    iter_count = 0
    while changed and iter_count < max_iter:
        changed = False
        iter_count += 1
        
        # Expand L: find w not in L∪R such that R ⊆ rem_adj[w]
        best_w_L = None
        best_score_L = -1
        for w in range(n):
            if w in L or w in R:
                continue
            if R.issubset(rem_adj[w]):
                # Score: prefer vertices connected to many vertices in R (already checked) 
                # and also have high degree overall
                score = len(rem_adj[w])
                if score > best_score_L:
                    best_score_L = score
                    best_w_L = w
        if best_w_L is not None:
            L.add(best_w_L)
            changed = True
        
        # Expand R: find w not in L∪R such that L ⊆ rem_adj[w]
        best_w_R = None
        best_score_R = -1
        for w in range(n):
            if w in L or w in R:
                continue
            if L.issubset(rem_adj[w]):
                score = len(rem_adj[w])
                if score > best_score_R:
                    best_score_R = score
                    best_w_R = w
        if best_w_R is not None:
            R.add(best_w_R)
            changed = True
    
    # Count removed edges
    removed_count = 0
    for l in L:
        for r in R:
            if frozenset((l, r)) in remaining_edges:
                removed_count += 1
    
    return L, R, removed_count

def _greedy_partition(remaining_edges, n, deadline, max_candidates=5):
    """Greedy biclique partition. Returns list of bicliques or None if timeout."""
    current_remaining = set(remaining_edges)
    bicliques = []
    
    while current_remaining:
        if time.time() > deadline:
            return None
        
        # Compute degrees
        deg = [0] * n
        for e in current_remaining:
            u, v = list(e)
            deg[u] += 1
            deg[v] += 1
        
        # Pick top candidates by degree
        sorted_verts = sorted(range(n), key=lambda x: deg[x], reverse=True)
        top_candidates = [v for v in sorted_verts[:max_candidates] if deg[v] > 0]
        
        if not top_candidates:
            break
        
        # Try each candidate, pick the one giving the largest biclique
        best_L = None
        best_R = None
        best_removed = 0
        
        for v_start in top_candidates:
            L, R, removed = _expand_biclique(v_start, current_remaining, n)
            if removed > best_removed:
                best_removed = removed
                best_L = L
                best_R = R
        
        if best_removed == 0 or best_L is None:
            # Fallback: single edge biclique
            e = next(iter(current_remaining))
            u, v = list(e)
            bicliques.append([[u], [v]])
            current_remaining.remove(e)
            continue
        
        # Remove cross edges
        for l in best_L:
            for r in best_R:
                e = frozenset((l, r))
                if e in current_remaining:
                    current_remaining.remove(e)
        
        bicliques.append([sorted(best_L), sorted(best_R)])
    
    return bicliques

def _try_merge_bicliques(bicliques, host_adj, n):
    """Try to merge compatible bicliques. Returns merged list."""
    if len(bicliques) <= 1:
        return bicliques
    
    merged = True
    while merged:
        merged = False
        new_bicliques = []
        i = 0
        while i < len(bicliques):
            if i >= len(bicliques):
                break
            L1, R1 = bicliques[i][0], bicliques[i][1]
            j = i + 1
            found_merge = False
            while j < len(bicliques):
                L2, R2 = bicliques[j][0], bicliques[j][1]
                
                # Try merging: new_L = L1 ∪ L2, new_R = R1 ∪ R2
                # Check if all cross pairs are host edges
                new_L = set(L1) | set(L2)
                new_R = set(R1) | set(R2)
                
                # Check: for every l in new_L and r in new_R, (l,r) must be a host edge
                # Also need: no vertex in new_L ∩ new_R
                if new_L & new_R:
                    j += 1
                    continue
                
                valid = True
                for l in new_L:
                    for r in new_R:
                        if r not in host_adj[l]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    new_bicliques.append([sorted(new_L), sorted(new_R)])
                    merged = True
                    found_merge = True
                    break
                j += 1
            
            if not found_merge:
                new_bicliques.append(bicliques[i])
            i += 1
        
        bicliques = new_bicliques
    
    return bicliques

def _validate_partition(bicliques, all_edges_set, n):
    """Validate that bicliques form a valid partition of all edges."""
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        for l in left:
            for r in right:
                e = frozenset((l, r))
                if e in covered:
                    return False
                covered.add(e)
                if e not in all_edges_set:
                    return False
    return covered == all_edges_set

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = P['n']
    
    # Build static adjacency sets for host graph
    host_adj = _build_host_adj(n, P['edges'])
    
    # Initialize remaining edges as a set of frozensets
    all_edges_set = set()
    for u, v in P['edges']:
        if u < v:
            all_edges_set.add(frozenset((u, v)))
        else:
            all_edges_set.add(frozenset((v, u)))
    
    best_solution = None
    best_count = float('inf')
    
    # Multiple attempts with different random seeds for candidate ordering
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        random.seed(attempt)
        
        # Shuffle candidate order for diversity
        current_remaining = set(all_edges_set)
        bicliques = []
        valid_solution = True
        
        while current_remaining:
            if time.time() > deadline:
                valid_solution = False
                break
            
            # Compute degrees
            deg = [0] * n
            for e in current_remaining:
                u, v = list(e)
                deg[u] += 1
                deg[v] += 1
            
            # Pick candidates: mix of high-degree and random
            nonzero_verts = [v for v in range(n) if deg[v] > 0]
            if not nonzero_verts:
                break
            
            # Sort by degree descending, take top few, add some random
            sorted_verts = sorted(nonzero_verts, key=lambda x: deg[x], reverse=True)
            top_k = min(8, len(sorted_verts))
            candidates = sorted_verts[:top_k]
            
            # Add a few random candidates for diversity
            if len(nonzero_verts) > top_k:
                extra = random.sample(nonzero_verts, min(4, len(nonzero_verts) - top_k))
                for e_v in extra:
                    if e_v not in candidates:
                        candidates.append(e_v)
            
            # Try each candidate, pick the one giving the largest biclique
            best_L = None
            best_R = None
            best_removed = 0
            
            for v_start in candidates:
                L, R, removed = _expand_biclique(v_start, current_remaining, n)
                if removed > best_removed:
                    best_removed = removed
                    best_L = L
                    best_R = R
            
            if best_removed == 0 or best_L is None:
                # Fallback: single edge
                e = next(iter(current_remaining))
                u, v = list(e)
                bicliques.append([[int(u)], [int(v)]])
                current_remaining.remove(e)
                continue
            
            # Remove cross edges
            for l in best_L:
                for r in best_R:
                    e = frozenset((l, r))
                    if e in current_remaining:
                        current_remaining.remove(e)
            
            bicliques.append([sorted(best_L), sorted(best_R)])
        
        if not valid_solution:
            continue
        
        # Try to merge compatible bicliques
        bicliques = _try_merge_bicliques(bicliques, host_adj, n)
        
        # Validate
        if _validate_partition(bicliques, all_edges_set, n):
            if len(bicliques) < best_count:
                best_count = len(bicliques)
                best_solution = bicliques
                if best_count <= 3:
                    break
    
    if best_solution is None:
        # Fallback: per-vertex star decomposition
        bicliques = []
        for u in range(n):
            R_u = [v for v in host_adj[u] if v > u]
            if R_u:
                bicliques.append([[u], sorted(R_u)])
        best_solution = bicliques
        best_count = len(bicliques)
    
    # Convert to required format
    final_result = []
    for bic in best_solution:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        final_result.append([left, right])
    
    return final_result

# EVOLVE-BLOCK-END
