import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _bit_count(x):
    return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

def _is_regular_subset(adj_bits, subset, mask):
    k = len(subset)
    if k < 6:
        return False
    target_deg = -1
    for v in subset:
        deg = _bit_count(adj_bits[v] & mask)
        if target_deg == -1:
            target_deg = deg
        elif deg != target_deg:
            return False
    return True

def _precompute_data(n):
    """
    Precompute all subsets of size 6 and 7, their masks, and edge-to-subset mappings.
    Also precompute for each edge the list of (subset_index, is_6) it affects.
    """
    subsets_6 = list(combinations(range(n), 6))
    subsets_7 = list(combinations(range(n), 7))
    
    masks_6 = []
    for subset in subsets_6:
        m = 0
        for v in subset:
            m |= (1 << v)
        masks_6.append(m)
    
    masks_7 = []
    for subset in subsets_7:
        m = 0
        for v in subset:
            m |= (1 << v)
        masks_7.append(m)
    
    # For each edge (u,v), store list of (subset_idx, is_6_flag)
    # is_6_flag = True means it's a 6-subset index in subsets_6
    edge_subsets = [[[] for _ in range(n)] for __ in range(n)]
    
    for idx, subset in enumerate(subsets_6):
        for i in range(6):
            u = subset[i]
            for j in range(i+1, 6):
                v = subset[j]
                edge_subsets[u][v].append((idx, True))
                edge_subsets[v][u].append((idx, True))
    
    for idx, subset in enumerate(subsets_7):
        for i in range(7):
            u = subset[i]
            for j in range(i+1, 7):
                v = subset[j]
                edge_subsets[u][v].append((idx, False))
                edge_subsets[v][u].append((idx, False))
    
    return subsets_6, subsets_7, masks_6, masks_7, edge_subsets

def _compute_all_validity(adj_bits, subsets_6, subsets_7, masks_6, masks_7):
    """Compute validity for all 6 and 7 subsets."""
    valid_6 = [False] * len(subsets_6)
    valid_7 = [False] * len(subsets_7)
    
    for i, subset in enumerate(subsets_6):
        valid_6[i] = _is_regular_subset(adj_bits, subset, masks_6[i])
    
    for i, subset in enumerate(subsets_7):
        valid_7[i] = _is_regular_subset(adj_bits, subset, masks_7[i])
    
    return valid_6, valid_7

def _score_edge_flip(adj_bits, u, v, subsets_6, subsets_7, masks_6, masks_7, valid_6, valid_7, edge_subsets):
    """
    Score an edge flip by counting how many violations it would resolve
    minus how many new violations it would create.
    Returns (net_resolution, flip_would_resolve_count, flip_would_create_count)
    """
    # Determine current state of edge
    has_edge = bool(adj_bits[u] & (1 << v))
    
    # Temporarily flip to check
    if has_edge:
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)
    else:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
    
    resolve_count = 0
    create_count = 0
    
    for (idx, is_6) in edge_subsets[u][v]:
        if is_6:
            subset = subsets_6[idx]
            mask = masks_6[idx]
            was_valid = valid_6[idx]
            now_valid = _is_regular_subset(adj_bits, subset, mask)
            if was_valid and not now_valid:
                resolve_count += 1
            elif not was_valid and now_valid:
                create_count += 1
        else:
            subset = subsets_7[idx]
            mask = masks_7[idx]
            was_valid = valid_7[idx]
            now_valid = _is_regular_subset(adj_bits, subset, mask)
            if was_valid and not now_valid:
                resolve_count += 1
            elif not was_valid and now_valid:
                create_count += 1
    
    # Restore edge
    if has_edge:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
    else:
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)
    
    return (resolve_count - create_count, resolve_count, create_count)

def _find_best_flip(adj_bits, subsets_6, subsets_7, masks_6, masks_7, valid_6, valid_7, edge_subsets, n):
    """
    Find the edge flip that maximizes net resolution of violations.
    Returns (u, v, score) or (None, None, -1) if no beneficial flip found.
    """
    best_score = -1
    best_u = None
    best_v = None
    
    for u in range(n):
        for v in range(u + 1, n):
            score, _, _ = _score_edge_flip(adj_bits, u, v, subsets_6, subsets_7, masks_6, masks_7, valid_6, valid_7, edge_subsets)
            if score > best_score:
                best_score = score
                best_u = u
                best_v = v
    
    return best_u, best_v, best_score

def _check_larger_violations(adj_bits, n, subsets_6, subsets_7, masks_6, masks_7, valid_6, valid_7):
    """
    Check if there are violations in subsets of size 8 or larger.
    Returns (violating_subset, mask) or (None, None).
    """
    for k in range(8, n + 1):
        for subset in combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if _is_regular_subset(adj_bits, subset, mask):
                return subset, mask
    return None, None

def construct():
    start_time = time.time()
    deadline = 145.0
    
    best_result = None
    
    for n_target in [21, 20]:
        if time.time() > deadline - 10.0:
            break
        
        # Precompute for this n
        subsets_6, subsets_7, masks_6, masks_7, edge_subsets = _precompute_data(n_target)
        
        max_restarts = 15
        found_valid = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
            
            # Initialize random graph with slight bias
            adj_bits = [0] * n_target
            p = 0.5
            for u in range(n_target):
                for v in range(u + 1, n_target):
                    if random.random() < p:
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
            
            # Compute initial validity
            valid_6, valid_7 = _compute_all_validity(adj_bits, subsets_6, subsets_7, masks_6, masks_7)
            
            # Local search with constraint propagation
            max_steps = 3000
            for step in range(max_steps):
                if time.time() > deadline - 1.0:
                    break
                
                # Count current violations
                viol_6_count = sum(1 for v in valid_6 if v)
                viol_7_count = sum(1 for v in valid_7 if v)
                
                if viol_6_count == 0 and viol_7_count == 0:
                    # Check larger sizes
                    viol_subset, viol_mask = _check_larger_violations(adj_bits, n_target, subsets_6, subsets_7, masks_6, masks_7, valid_6, valid_7)
                    
                    if viol_subset is None:
                        # Valid graph!
                        edges = []
                        for u in range(n_target):
                            for v in range(u + 1, n_target):
                                if adj_bits[u] & (1 << v):
                                    edges.append([u, v])
                        
                        if best_result is None or n_target > best_result[0]:
                            best_result = (n_target, edges)
                        found_valid = True
                        break
                    else:
                        # Found larger violation - flip a random edge in it
                        u, v = random.sample(list(viol_subset), 2)
                        if adj_bits[u] & (1 << v):
                            adj_bits[u] &= ~(1 << v)
                            adj_bits[v] &= ~(1 << u)
                        else:
                            adj_bits[u] |= (1 << v)
                            adj_bits[v] |= (1 << u)
                        
                        # Update validity for affected subsets
                        for (idx, is_6) in edge_subsets[u][v]:
                            if is_6:
                                valid_6[idx] = _is_regular_subset(adj_bits, subsets_6[idx], masks_6[idx])
                            else:
                                valid_7[idx] = _is_regular_subset(adj_bits, subsets_7[idx], masks_7[idx])
                        continue
                
                # Find the best edge flip using constraint propagation scoring
                best_u, best_v, best_score = _find_best_flip(adj_bits, subsets_6, subsets_7, masks_6, masks_7, valid_6, valid_7, edge_subsets, n_target)
                
                if best_u is None or best_score <= 0:
                    # No beneficial flip found - try a random flip to escape local minimum
                    u = random.randint(0, n_target - 1)
                    v = random.randint(0, n_target - 1)
                    if u == v:
                        v = (v + 1) % n_target
                    if u > v:
                        u, v = v, u
                    
                    if adj_bits[u] & (1 << v):
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
                    else:
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
                else:
                    # Perform the best flip
                    u, v = best_u, best_v
                    if adj_bits[u] & (1 << v):
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
                    else:
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
                
                # Update validity for affected subsets
                for (idx, is_6) in edge_subsets[u][v]:
                    if is_6:
                        valid_6[idx] = _is_regular_subset(adj_bits, subsets_6[idx], masks_6[idx])
                    else:
                        valid_7[idx] = _is_regular_subset(adj_bits, subsets_7[idx], masks_7[idx])
        
        if found_valid:
            break
    
    if best_result is not None:
        n_found, edges_found = best_result
        return {'n': n_found, 'edges': edges_found}
    else:
        return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
