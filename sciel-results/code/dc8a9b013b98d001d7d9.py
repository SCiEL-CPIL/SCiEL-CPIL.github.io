import time
import random
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)


def _expand_biclique(seed_u, seed_v, remaining_edges, adj_matrix, n):
    """
    Expand a biclique starting from edge (seed_u, seed_v) using only edges in remaining_edges.
    Returns (L, R, cross_edge_set) or None if invalid.
    """
    L = {seed_u}
    R = {seed_v}
    
    # Verify the seed edge exists
    if _edge_key(seed_u, seed_v) not in remaining_edges:
        return None
    
    changed = True
    max_iters = 200
    iters = 0
    while changed and iters < max_iters:
        changed = False
        iters += 1
        
        # Try adding to L: w can join L if for all x in R, (w,x) in remaining_edges
        for w in range(n):
            if w in L or w in R:
                continue
            can_add = True
            for x in R:
                if _edge_key(w, x) not in remaining_edges:
                    can_add = False
                    break
            if can_add:
                L.add(w)
                changed = True
        
        # Try adding to R: w can join R if for all x in L, (w,x) in remaining_edges
        for w in range(n):
            if w in L or w in R:
                continue
            can_add = True
            for x in L:
                if _edge_key(w, x) not in remaining_edges:
                    can_add = False
                    break
            if can_add:
                R.add(w)
                changed = True
    
    # Compute cross edges and verify
    cross_edges = set()
    for u in L:
        for v in R:
            ek = _edge_key(u, v)
            if ek not in remaining_edges:
                return None
            cross_edges.add(ek)
    
    if len(cross_edges) == 0:
        return None
    
    return (frozenset(L), frozenset(R), cross_edges)


def _greedy_partition(remaining_edges, adj_matrix, n, deadline, rng, seed_priority='degree'):
    """
    Iterative greedy: at each step, find the best biclique among remaining edges,
    add it to the partition, remove its edges.
    """
    bicliques = []
    remaining = set(remaining_edges)
    
    while remaining:
        if time.time() > deadline - 2.0:
            break
        
        # Generate candidate seed edges
        candidates = list(remaining)
        
        # Sort by a heuristic: degree sum of endpoints
        if seed_priority == 'degree':
            candidates.sort(key=lambda e: -(len(adj_matrix[e[0]]) + len(adj_matrix[e[1]])))
        elif seed_priority == 'random':
            rng.shuffle(candidates)
        elif seed_priority == 'mixed':
            # Take top by degree but shuffle within top-K
            candidates.sort(key=lambda e: -(len(adj_matrix[e[0]]) + len(adj_matrix[e[1]])))
            k = min(20, len(candidates))
            top = candidates[:k]
            rng.shuffle(top)
            candidates = top + candidates[k:]
        
        # Try a limited number of seeds to find the best biclique
        max_seeds = min(15, len(candidates))
        best_biclique = None
        best_count = 0
        
        for i in range(max_seeds):
            if time.time() > deadline - 1.0:
                break
            u, v = candidates[i]
            result = _expand_biclique(u, v, remaining, adj_matrix, n)
            if result is None:
                continue
            L, R, cross = result
            cnt = len(cross)
            if cnt > best_count:
                best_count = cnt
                best_biclique = (L, R, cross)
        
        if best_biclique is None:
            # Fallback: take first remaining edge
            u, v = candidates[0]
            bicliques.append((frozenset({u}), frozenset({v})))
            remaining.discard(_edge_key(u, v))
        else:
            L, R, cross = best_biclique
            bicliques.append((L, R))
            remaining -= cross
    
    return bicliques, remaining


def _try_merge(bicliques, remaining_edges, adj_matrix, n, deadline):
    """
    Try to merge pairs of bicliques if their union forms a valid biclique.
    Returns the merged list.
    """
    current = [(list(L), list(R)) for L, R in bicliques]
    merged = True
    pass_count = 0
    
    while merged and pass_count < 10:
        if time.time() > deadline - 1.0:
            break
        merged = False
        pass_count += 1
        
        i = 0
        while i < len(current):
            if time.time() > deadline - 1.0:
                break
            j = i + 1
            while j < len(current):
                if time.time() > deadline - 1.0:
                    break
                
                Li, Ri = current[i]
                Lj, Rj = current[j]
                
                # Try all 4 combinations of union
                candidates = [
                    (set(Li) | set(Lj), set(Ri) | set(Rj)),
                    (set(Li) | set(Rj), set(Ri) | set(Lj)),
                    (set(Ri) | set(Lj), set(Li) | set(Rj)),
                    (set(Ri) | set(Rj), set(Li) | set(Lj)),
                ]
                
                best_union = None
                for Lc, Rc in candidates:
                    if not Lc or not Rc:
                        continue
                    # Check if all cross edges exist in the original graph
                    valid = True
                    cross_set = set()
                    for u in Lc:
                        for v in Rc:
                            ek = _edge_key(u, v)
                            if ek not in remaining_edges:
                                valid = False
                                break
                            cross_set.add(ek)
                        if not valid:
                            break
                    if valid and len(cross_set) > 0:
                        best_union = (Lc, Rc, cross_set)
                        break
                
                if best_union is not None:
                    Lc, Rc, _ = best_union
                    current[i] = (list(Lc), list(Rc))
                    del current[j]
                    merged = True
                    # Don't increment j, re-check same i
                    continue
                j += 1
            i += 1
    
    return current


def _local_search(bicliques, remaining_edges, adj_matrix, n, deadline, rng):
    """
    Try to improve by replacing a biclique with a larger one that covers more edges.
    Also try splitting a biclique if it leads to better coverage.
    """
    current = [(list(L), list(R)) for L, R in bicliques]
    
    # Compute the edge set covered by current bicliques
    covered_edges = set()
    for L, R in current:
        for u in L:
            for v in R:
                covered_edges.add(_edge_key(u, v))
    
    # Try to find a better biclique for the "worst" biclique (smallest one)
    if len(current) <= 1:
        return current
    
    # Find the smallest biclique
    min_idx = 0
    min_size = len(current[0][0]) * len(current[0][1])
    for i in range(1, len(current)):
        size = len(current[i][0]) * len(current[i][1])
        if size < min_size:
            min_size = size
            min_idx = i
    
    # Get edges covered by the smallest biclique
    L_small, R_small = current[min_idx]
    small_edges = set()
    for u in L_small:
        for v in R_small:
            small_edges.add(_edge_key(u, v))
    
    # Try to find a biclique that covers small_edges plus more
    # Remove small_edges from consideration, find best biclique covering them
    other_edges = covered_edges - small_edges
    
    # Try expanding from the small biclique's edges
    best_new = None
    best_new_count = 0
    
    for u, v in list(small_edges)[:10]:
        result = _expand_biclique(u, v, covered_edges, adj_matrix, n)
        if result is None:
            continue
        L, R, cross = result
        # Check if this new biclique covers all of small_edges
        if small_edges.issubset(cross):
            cnt = len(cross)
            if cnt > best_new_count:
                best_new_count = cnt
                best_new = (L, R, cross)
    
    if best_new is not None:
        L, R, cross = best_new
        # Replace the small biclique and remove its edges from others
        new_current = []
        for i, (Li, Ri) in enumerate(current):
            if i == min_idx:
                continue
            # Remove edges that are in cross
            new_L = set(Li)
            new_R = set(Ri)
            # We need to re-validate: if an edge (u,v) in this biclique is in cross,
            # we can't keep it. But since bicliques must partition edges, we need
            # to remove overlapping edges. This is complex; skip for now.
            new_current.append((list(new_L), list(new_R)))
        
        # Actually, let's just try a simpler approach: replace the small one
        # and re-partition the rest
        # This is too complex; just return current
        pass
    
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj_matrix = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj_matrix[u].add(v)
        adj_matrix[v].add(u)
        edge_set.add(_edge_key(u, v))
    
    # Precompute degree
    deg = [len(adj_matrix[i]) for i in range(n)]
    
    best_bicliques = None
    best_count = float('inf')
    
    # Multiple restarts with different strategies
    num_restarts = 25
    per_restart_time = min(3.0, (deadline - start_time - 10.0) / max(1, num_restarts))
    
    for restart in range(num_restarts):
        if time.time() > deadline - 8.0:
            break
        
        rng = random.Random(restart * 7919 + 13)
        
        # Choose strategy for this restart
        if restart % 3 == 0:
            priority = 'degree'
        elif restart % 3 == 1:
            priority = 'random'
        else:
            priority = 'mixed'
        
        restart_deadline = min(deadline - 5.0, start_time + (restart + 1) * per_restart_time + 5.0)
        
        bicliques, remaining = _greedy_partition(
            edge_set, adj_matrix, n, restart_deadline, rng, priority
        )
        
        # Handle any remaining edges
        if remaining:
            for u, v in remaining:
                bicliques.append((frozenset({u}), frozenset({v})))
        
        # Post-processing: try to merge
        merged = _try_merge(bicliques, edge_set, adj_matrix, n, restart_deadline)
        
        # Convert to list format
        result = [[list(L), list(R)] for L, R in merged]
        
        cnt = len(result)
        if cnt < best_count:
            best_count = cnt
            best_bicliques = result
        
        if best_count <= 5:
            break
    
    # If we still have time, do more restarts with tighter time
    if best_bicliques is not None and time.time() < deadline - 15.0:
        extra_restarts = 10
        for extra in range(extra_restarts):
            if time.time() > deadline - 5.0:
                break
            
            rng = random.Random(100000 + extra * 31)
            restart_deadline = deadline - 3.0
            
            bicliques, remaining = _greedy_partition(
                edge_set, adj_matrix, n, restart_deadline, rng, 'mixed'
            )
            
            if remaining:
                for u, v in remaining:
                    bicliques.append((frozenset({u}), frozenset({v})))
            
            merged = _try_merge(bicliques, edge_set, adj_matrix, n, restart_deadline)
            result = [[list(L), list(R)] for L, R in merged]
            
            cnt = len(result)
            if cnt < best_count:
                best_count = cnt
                best_bicliques = result
    
    # Final validation and type conversion
    if best_bicliques is None:
        # Fallback: one biclique per edge
        best_bicliques = [[list([u]), list([v])] for u, v in edges]
    
    # Ensure all types are correct: integers as int, lists as list
    final_result = []
    for biclique in best_bicliques:
        L = [int(x) for x in biclique[0]]
        R = [int(x) for x in biclique[1]]
        if L and R:
            final_result.append([L, R])
    
    return final_result


if __name__ == '__main__':
    result = construct()
    print(f"Number of bicliques: {len(result)}")
    # Validate
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add(_edge_key(u, v))
    
    covered = set()
    for biclique in result:
        L, R = biclique
        for u in L:
            for v in R:
                ek = _edge_key(u, v)
                if ek in covered:
                    print(f"OVERLAP: {ek}")
                covered.add(ek)
    
    missing = edge_set - covered
    extra = covered - edge_set
    if missing:
        print(f"Missing {len(missing)} edges")
    if extra:
        print(f"Extra {len(extra)} edges")
    print(f"Total edges: {len(edge_set)}, covered: {len(covered)}")
