import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    # Use 80s limit to be safe within 90s kill
    deadline = start_time + 80.0
    
    n = P['n']
    edges = P['edges']
    
    # Map edges to indices
    edge_to_idx = {}
    idx_to_edge = []
    for i, (u, v) in enumerate(edges):
        if u > v:
            u, v = v, u
        edge_to_idx[(u, v)] = i
        idx_to_edge.append((u, v))
        
    m = len(edges)
    
    # Adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Generate maximal bicliques
    candidates = []
    seen_bicliques = set()
    
    # To speed up, we can iterate over edges and expand
    for u, v in edges:
        if time.time() > deadline - 10.0: # Reserve time for B&B
            break
            
        L = {u}
        R = {v}
        
        changed = True
        while changed:
            changed = False
            
            # Expand L
            new_L = set(L)
            for x in range(n):
                if x in R or x in new_L:
                    continue
                # Check if x is connected to all in R
                if all((adj[x] & (1 << y)) for y in R):
                    new_L.add(x)
            if new_L != L:
                L = new_L
                changed = True
                
            # Expand R
            new_R = set(R)
            for y in range(n):
                if y in L or y in new_R:
                    continue
                # Check if y is connected to all in L
                if all((adj[x] & (1 << y)) for x in L):
                    new_R.add(y)
            if new_R != R:
                R = new_R
                changed = True
                
        # Normalize and check uniqueness
        L_frozen = frozenset(L)
        R_frozen = frozenset(R)
        key = (L_frozen, R_frozen)
        if key not in seen_bicliques:
            seen_bicliques.add(key)
            
            # Calculate edge mask for this biclique
            mask = 0
            for uu in L:
                for vv in R:
                    if uu == vv:
                        continue
                    key_e = (uu, vv) if uu < vv else (vv, uu)
                    if key_e in edge_to_idx:
                        mask |= (1 << edge_to_idx[key_e])
            
            if mask != 0:
                candidates.append((mask, list(L), list(R)))

    # Deduplicate masks (keep first occurrence)
    unique_candidates = []
    seen_masks = set()
    for mask, L, R in candidates:
        if mask not in seen_masks:
            seen_masks.add(mask)
            unique_candidates.append((mask, L, R))
            
    if not unique_candidates:
        # Fallback: single edges
        return [[ [u], [v] ] for u, v in edges]

    # Prepare for B&B
    universe_mask = (1 << m) - 1
    
    # Sort candidates by size (larger first) to help pruning
    unique_candidates.sort(key=lambda x: bin(x[0]).count('1'), reverse=True)
    
    # Precompute edge_sets: for each edge, which candidate indices cover it
    edge_sets = [[] for _ in range(m)]
    for i, (mask, L, R) in enumerate(unique_candidates):
        m_temp = mask
        while m_temp:
            lsb = m_temp & -m_temp
            idx = lsb.bit_length() - 1
            edge_sets[idx].append(i)
            m_temp ^= lsb
            
    max_set_size = max([bin(c[0]).count('1') for c in unique_candidates]) if unique_candidates else 1
    
    best_cover_indices = []
    best_size = m # Upper bound
    
    # B&B
    def bb(uncovered, current_size, current_cover):
        nonlocal best_size, best_cover_indices
        
        if time.time() > deadline:
            return
            
        if current_size >= best_size:
            return
            
        if uncovered == 0:
            if current_size < best_size:
                best_size = current_size
                best_cover_indices = current_cover[:]
            return
            
        # Lower bound
        # Simple LB: ceil(popcount(uncovered) / max_set_size)
        lb = (bin(uncovered).count('1') + max_set_size - 1) // max_set_size
        if current_size + lb >= best_size:
            return
            
        # Select edge with fewest candidates
        u_temp = uncovered
        best_edge = -1
        min_sets = 10000
        while u_temp:
            lsb = u_temp & -u_temp
            idx = lsb.bit_length() - 1
            if len(edge_sets[idx]) < min_sets:
                min_sets = len(edge_sets[idx])
                best_edge = idx
            u_temp ^= lsb
            
        if best_edge == -1:
            return
            
        # Branch on candidates covering best_edge
        # Sort by coverage in uncovered
        sets_to_try = edge_sets[best_edge]
        sets_to_try.sort(key=lambda i: bin(uncovered & unique_candidates[i][0]).count('1'), reverse=True)
        
        for i in sets_to_try:
            mask = unique_candidates[i][0]
            new_uncovered = uncovered & ~mask
            current_cover.append(i)
            bb(new_uncovered, current_size + 1, current_cover)
            current_cover.pop()
            
            if time.time() > deadline:
                return

    bb(universe_mask, 0, [])
    
    # Construct result
    if best_cover_indices:
        result = []
        for idx in best_cover_indices:
            mask, L, R = unique_candidates[idx]
            result.append([L, R])
            
        # Validate result
        covered_edges = set()
        valid = True
        for L, R in result:
            for u in L:
                for v in R:
                    if u == v: continue
                    key = (u, v) if u < v else (v, u)
                    if key not in edge_to_idx:
                        valid = False
                        break
                    if key in covered_edges:
                        valid = False
                        break
                    covered_edges.add(key)
            if not valid: break
        
        if valid and len(covered_edges) == m:
            return result
        else:
            # Invalid or incomplete, fallback
            return [[ [u], [v] ] for u, v in edges]
    else:
        # No solution found, fallback
        return [[ [u], [v] ] for u, v in edges]
