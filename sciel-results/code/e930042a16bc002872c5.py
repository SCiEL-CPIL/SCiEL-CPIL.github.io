import time
import random
import numpy as np
from itertools import combinations

def _edges_from_adj_masks(adj_masks, n):
    edges = []
    for i in range(n):
        m = adj_masks[i]
        for j in range(i + 1, n):
            if (m >> j) & 1:
                edges.append([i, j])
    return edges

def _find_regular_subgraph_fast(adj_masks, n, max_size=None):
    """
    Find a regular induced subgraph of size >= 6.
    Returns (mask, size) or (None, None) if none found.
    Checks smaller sizes first for faster rejection.
    """
    if max_size is None:
        max_size = n
    
    # Check sizes from 6 to max_size
    for k in range(6, max_size + 1):
        # For each k-subset, check if regular
        # Use combinations for clarity, but optimize inner loop
        for verts in combinations(range(n), k):
            v0 = verts[0]
            mask = 0
            for v in verts:
                mask |= (1 << v)
            
            first_deg = (adj_masks[v0] & mask).bit_count()
            
            # Parity check
            if (k * first_deg) % 2 != 0:
                continue
            
            is_reg = True
            for v in verts[1:]:
                if (adj_masks[v] & mask).bit_count() != first_deg:
                    is_reg = False
                    break
            
            if is_reg:
                return mask, k
    
    return None, None

def _count_regular_subgraphs(adj_masks, n, limit=10):
    """Count regular subgraphs up to a limit, for fitness."""
    count = 0
    for k in range(6, n + 1):
        for verts in combinations(range(n), k):
            v0 = verts[0]
            mask = 0
            for v in verts:
                mask |= (1 << v)
            first_deg = (adj_masks[v0] & mask).bit_count()
            if (k * first_deg) % 2 != 0:
                continue
            is_reg = True
            for v in verts[1:]:
                if (adj_masks[v] & mask).bit_count() != first_deg:
                    is_reg = False
                    break
            if is_reg:
                count += 1
                if count >= limit:
                    return count
    return count

def _local_search(adj_masks, n, max_iters=200, deadline=None):
    """
    Local search: find regular subgraph, flip edge in it to break it.
    Returns (adj_masks, is_valid).
    """
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        mask, k = _find_regular_subgraph_fast(adj_masks, n)
        
        if mask is None:
            return adj_masks, True
        
        # Extract vertices in the regular subgraph
        verts = []
        m = mask
        while m:
            lsb = m & -m
            verts.append(lsb.bit_length() - 1)
            m -= lsb
        
        # Try flipping edges within the subgraph or incident to it
        # Prioritize edges within the subgraph
        candidates = []
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                candidates.append((verts[i], verts[j]))
        
        # Also add some edges from subgraph to outside
        for v in verts:
            for u in range(n):
                if u not in verts:
                    candidates.append((min(v, u), max(v, u)))
        
        random.shuffle(candidates)
        
        improved = False
        for (i, j) in candidates[:20]:  # Try up to 20 flips
            if deadline and time.time() > deadline:
                break
            # Flip edge (i, j)
            adj_masks[i] ^= (1 << j)
            adj_masks[j] ^= (1 << i)
            
            # Quick check: did we break the specific subgraph?
            # Re-check if this exact mask is still regular
            v0 = verts[0]
            first_deg = (adj_masks[v0] & mask).bit_count()
            still_reg = True
            for v in verts[1:]:
                if (adj_masks[v] & mask).bit_count() != first_deg:
                    still_reg = False
                    break
            
            if not still_reg:
                # Good, we broke it. But check if we created a new one
                # Do a quick full check
                new_mask, new_k = _find_regular_subgraph_fast(adj_masks, n)
                if new_mask is None:
                    return adj_masks, True
                # Otherwise, undo and try another flip
                adj_masks[i] ^= (1 << j)
                adj_masks[j] ^= (1 << i)
                improved = True
                break
            else:
                # Undo
                adj_masks[i] ^= (1 << j)
                adj_masks[j] ^= (1 << i)
        
        if not improved:
            # Try a random flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                if i > j:
                    i, j = j, i
                adj_masks[i] ^= (1 << j)
                adj_masks[j] ^= (1 << i)
    
    # Final check
    mask, k = _find_regular_subgraph_fast(adj_masks, n)
    return adj_masks, (mask is None)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_n = 0
    best_adj_masks = []
    
    # Try n=22, 21, 20
    for target_n in [22, 21, 20]:
        if time.time() > deadline - 10:
            break
        
        n = target_n
        found = False
        
        # Multiple restarts with local search
        num_restarts = 30 if n >= 21 else 15
        max_iters = 300 if n >= 21 else 200
        
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Generate random graph
            adj_masks = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            # Local search
            adj_masks, is_valid = _local_search(adj_masks, n, max_iters=max_iters, deadline=deadline - 5)
            
            if is_valid:
                best_n = n
                best_adj_masks = adj_masks
                found = True
                break
        
        if found:
            break
    
    if best_n == 0:
        # Fallback: try smaller n with more aggressive search
        for target_n in range(19, 4, -1):
            if time.time() > deadline - 3:
                break
            n = target_n
            for r in range(20):
                if time.time() > deadline - 2:
                    break
                adj_masks = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj_masks[i] |= (1 << j)
                            adj_masks[j] |= (1 << i)
                
                adj_masks, is_valid = _local_search(adj_masks, n, max_iters=100, deadline=deadline - 2)
                
                if is_valid:
                    best_n = n
                    best_adj_masks = adj_masks
                    break
            if best_n > 0:
                break
    
    if best_n == 0:
        best_n = 5
        best_adj_masks = [0] * 5
    
    edges = _edges_from_adj_masks(best_adj_masks, best_n)
    return {'n': best_n, 'edges': edges}
