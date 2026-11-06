"""Improved program for R(K2,2,2) maximization via guided local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _find_k222(adj_mask, n):
    """
    Find a K2,2,2 subgraph in the graph defined by adj_mask.
    Returns a tuple of 6 vertices (a,b,c,d,e,f) representing the K2,2,2, or None if not found.
    K2,2,2 is defined by 3 disjoint non-edges (a,b), (c,d), (e,f) such that all other pairs are edges.
    """
    if n < 6:
        return None
    
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if not (adj_mask[i] >> j) & 1:
                non_edges.append((i, j))
    
    if len(non_edges) < 3:
        return None
        
    len_ne = len(non_edges)
    for i1 in range(len_ne):
        a, b = non_edges[i1]
        common_ab = adj_mask[a] & adj_mask[b]
        common_ab &= ~(1 << a) & ~(1 << b)
        
        for i2 in range(i1 + 1, len_ne):
            c, d = non_edges[i2]
            if (c == a or c == b or d == a or d == b):
                continue
            
            if not ((common_ab >> c) & 1) or not ((common_ab >> d) & 1):
                continue
            
            candidates = common_ab & adj_mask[c] & adj_mask[d]
            candidates &= ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
            
            m = candidates
            candidate_list = []
            while m:
                lsb = m & -m
                idx = (lsb.bit_length() - 1)
                candidate_list.append(idx)
                m ^= lsb
            
            len_cl = len(candidate_list)
            for k1 in range(len_cl):
                e = candidate_list[k1]
                for k2 in range(k1 + 1, len_cl):
                    f = candidate_list[k2]
                    if not ((adj_mask[e] >> f) & 1):
                        return (a, b, c, d, e, f)
                        
    return None

def _adj_to_mask(adj, n):
    """Convert numpy boolean adjacency matrix to list of bitmasks."""
    masks = [0] * n
    for i in range(n):
        mask = 0
        row = adj[i]
        for j in range(n):
            if row[j]:
                mask |= (1 << j)
        masks[i] = mask
    return masks

def _is_valid(adj, n):
    """Check that neither color class contains K2,2,2."""
    adj_mask_0 = _adj_to_mask(adj, n)
    if _find_k222(adj_mask_0, n) is not None:
        return False
    
    adj_mask_1 = [0] * n
    all_ones = (1 << n) - 1
    for i in range(n):
        adj_mask_1[i] = (~adj_mask_0[i]) & all_ones & ~(1 << i)
        
    if _find_k222(adj_mask_1, n) is not None:
        return False
        
    return True

def _guided_repair(adj, n, deadline):
    """
    Guided local search: flip edges involved in current violations.
    """
    adj = adj.copy()
    masks_0 = _adj_to_mask(adj, n)
    all_ones = (1 << n) - 1
    
    def get_masks_1():
        masks_1 = [0] * n
        for i in range(n):
            masks_1[i] = (~masks_0[i]) & all_ones & ~(1 << i)
        return masks_1

    # Initial check
    viol0 = _find_k222(masks_0, n)
    masks_1 = get_masks_1()
    viol1 = _find_k222(masks_1, n)
    
    if viol0 is None and viol1 is None:
        return adj

    steps = 0
    max_steps = 50000
    while steps < max_steps and time.time() < deadline:
        steps += 1
        
        # Decide which violation to fix
        # Randomly pick from available violations
        if viol0 is not None and viol1 is not None:
            if random.random() < 0.5:
                target_viol = viol0
                target_mask = masks_0
                is_mask0 = True
            else:
                target_viol = viol1
                target_mask = masks_1
                is_mask0 = False
        elif viol0 is not None:
            target_viol = viol0
            target_mask = masks_0
            is_mask0 = True
        else:
            target_viol = viol1
            target_mask = masks_1
            is_mask0 = False
            
        # Pick a random edge from the violation's 6 vertices
        verts = list(target_viol)
        random.shuffle(verts)
        # Pick a pair from these 6 vertices that is a "non-edge" in the target graph?
        # No, we just pick any edge among the 6 vertices and flip it.
        # Flipping an edge that is part of the "missing" structure is best.
        # In K2,2,2, the non-edges are (a,b), (c,d), (e,f).
        # The edges are the other 12.
        # If we flip a "non-edge" to an edge, we break the K2,2,2.
        # If we flip an "edge" to a non-edge, we might create more problems.
        # So we should prefer flipping the 3 non-edges.
        
        # Identify the 3 non-edges in the violation
        a, b, c, d, e, f = target_viol
        non_edge_pairs = [(a, b), (c, d), (e, f)]
        
        # Pick one of these non-edges to flip
        ne_pair = random.choice(non_edge_pairs)
        i, j = ne_pair
        if i > j: i, j = j, i
        
        # Flip edge (i, j) in adj
        # Update masks
        bit_j = 1 << j
        bit_i = 1 << i
        
        # Determine current state in masks_0
        is_edge_0 = bool(masks_0[i] & bit_j)
        
        # Flip in masks_0
        if is_edge_0:
            masks_0[i] &= ~bit_j
            masks_0[j] &= ~bit_i
        else:
            masks_0[i] |= bit_j
            masks_0[j] |= bit_i
            
        # Update adj
        adj[i][j] = not is_edge_0
        adj[j][i] = not is_edge_0
        
        # Re-check violations
        # Only need to re-check if we flipped in mask0, mask1 changes automatically
        # But we need to re-detect if the specific violation is gone and if new ones appeared
        viol0 = _find_k222(masks_0, n)
        masks_1 = get_masks_1()
        viol1 = _find_k222(masks_1, n)
        
        if viol0 is None and viol1 is None:
            return adj
            
        # Optional: if stuck for a while, do a random flip to escape local minimum
        if steps % 100 == 0:
             i = random.randint(0, n-1)
             j = random.randint(0, n-1)
             if i == j: continue
             if i > j: i, j = j, i
             
             is_edge_0 = bool(masks_0[i] & bit_j)
             if is_edge_0:
                 masks_0[i] &= ~bit_j
                 masks_0[j] &= ~bit_i
             else:
                 masks_0[i] |= bit_j
                 masks_0[j] |= bit_i
             adj[i][j] = not is_edge_0
             adj[j][i] = not is_edge_0
             
             viol0 = _find_k222(masks_0, n)
             masks_1 = get_masks_1()
             viol1 = _find_k222(masks_1, n)

    return adj

def _paley_coloring(p):
    """Paley coloring of K_p where p is prime, p ≡ 1 mod 4."""
    if p % 4 != 1:
        return None
    squares = set()
    for x in range(1, p):
        squares.add(x*x % p)
    colors = []
    for i in range(p):
        for j in range(i+1, p):
            diff = (j - i) % p
            colors.append(0 if diff in squares else 1)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_n = 1
    best_colors = [0]
    
    # Try to get a good start from Paley
    start_colors = None
    for p in [17, 29, 13, 5]:
        if p % 4 != 1:
            continue
        cols = _paley_coloring(p)
        if cols:
            n_test = p
            adj_test = np.zeros((n_test, n_test), dtype=bool)
            idx = 0
            for i in range(n_test):
                for j in range(i+1, n_test):
                    adj_test[i][j] = (cols[idx] == 0)
                    adj_test[j][i] = (cols[idx] == 0)
                    idx += 1
            if _is_valid(adj_test, n_test):
                start_colors = cols
                best_n = n_test
                best_colors = cols
                break
                
    if start_colors is None:
        # Fallback to small valid graph
        n_test = 5
        adj_test = np.random.rand(n_test, n_test) > 0.5
        np.fill_diagonal(adj_test, False)
        adj_test = (adj_test + adj_test.T) > 0
        if _is_valid(adj_test, n_test):
            best_n = n_test
            best_colors = []
            for i in range(n_test):
                for j in range(i+1, n_test):
                    best_colors.append(1 if adj_test[i][j] else 0)
        else:
            best_n = 1
            best_colors = []

    # Try to extend best_n
    current_n = best_n
    current_adj = np.zeros((current_n, current_n), dtype=bool)
    idx = 0
    for i in range(current_n):
        for j in range(i+1, current_n):
            current_adj[i][j] = (best_colors[idx] == 0)
            current_adj[j][i] = (best_colors[idx] == 0)
            idx += 1

    for target_n in range(current_n + 1, 31):
        if time.time() > deadline:
            break
            
        n = target_n
        adj_new = np.zeros((n, n), dtype=bool)
        
        # Copy old
        adj_new[:current_n, :current_n] = current_adj
        
        # Initialize new edges
        for i in range(current_n, n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj_new[i][j] = True
                    adj_new[j][i] = True
        for i in range(current_n):
            for j in range(current_n, n):
                if random.random() < 0.5:
                    adj_new[i][j] = True
                    adj_new[j][i] = True
        
        # Guided repair
        adj_fixed = _guided_repair(adj_new, n, deadline)
        
        if _is_valid(adj_fixed, n):
            current_n = n
            current_adj = adj_fixed
            best_n = n
            best_colors = []
            for i in range(n):
                for j in range(i+1, n):
                    best_colors.append(0 if adj_fixed[i][j] else 1)

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
