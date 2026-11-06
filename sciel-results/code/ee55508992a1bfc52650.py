import time
import random
import math
import numpy as np

def construct():
    deadline = time.time() + 110  # Safety margin for 120s limit (or 150s if that's the hard kill)
    # The prompt says "evaluator KILLS after 150s", but also "finish within 120s".
    # I will use 140s as a safe internal deadline to allow for cleanup.
    
    # Target sizes: 26 is the known bound. 27 is the target for improvement.
    # We will try 27 first with a limited time budget, then 26, then 25...
    # If 27 is not found quickly, drop to 26.
    
    target_sizes = [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]
    
    best_n = 11
    # Fallback valid coloring for n=11 (from seed logic or simple random search)
    # The seed had a specific formula: [min(j - i, 11 - j + i) - 1 ...]
    # Let's generate a valid one for n=11 quickly if all else fails.
    # Actually, we can just run the solver for n=11 if we fail on larger ones.
    
    for n in target_sizes:
        if time.time() > deadline - 5:
            break
            
        # Allocate time based on n
        # Give more time to 26 and 27
        if n >= 26:
            time_limit = min(deadline - time.time() - 2, 30) # 30s for 27/26
        else:
            time_limit = min(deadline - time.time() - 2, 10)
            
        if time_limit < 1:
            continue
            
        success, colors = _fast_local_search(n, time.time() + time_limit)
        
        if success:
            best_n = n
            # We found a valid coloring for this n.
            # If n < 26, we can return immediately? 
            # The task says "Maximize n".
            # If we find 26, we should try 27? 
            # The loop goes 27 -> 26 -> 25.
            # If we find 27, we return.
            # If we find 26, we continue to 25? No, 26 is better than 25.
            # But we want to *maximize*.
            # So if we find 27, we are done.
            # If we find 26, we are done (unless we try 27 again? No, 27 was tried first).
            # Wait, the loop order is 27, 26, 25...
            # If 27 fails, we try 26. If 26 succeeds, we have a score of 26/11.
            # We should return 26.
            # We don't need to try 25.
            break
    
    if best_n == 11:
        # Fallback: generate a valid coloring for n=11
        # Simple random search for n=11 is very fast
        success, colors = _fast_local_search(11, time.time() + 5)
        if success:
            best_n = 11
            # Use the found colors
            # Need to return the colors list
            # The loop above doesn't save colors if best_n stays 11 initially unless we capture it.
            # Let's restructure to always return a valid coloring.
            pass
        else:
            # Hardcode a known valid coloring for n=11 or generate one deterministically
            # For n=11, a random search will almost certainly find one instantly.
            # Let's just force a search for n=11 if we haven't found anything better.
            # Actually, the loop above *does* try 17, 18... down to 11? 
            # No, the list is [27, 26, ..., 17].
            # If all fail, best_n is still 11 (initial value) but best_colors is the seed one.
            # The seed colors are valid for n=11.
            pass

    # Ensure we return a valid result
    # If best_n is 11, we use the seed colors.
    # If best_n > 11, we use the colors found in the loop.
    # We need to track the colors found.
    
    # Let's rewrite the loop logic to be cleaner.
    
    return {'n': best_n, 'colors': best_colors}

def _fast_local_search(n, deadline):
    """
    Fast local search with incremental violation counting.
    Returns (success, colors_list)
    """
    if n < 3:
        return True, []
        
    m = n * (n - 1) // 2
    colors = [random.randint(0, 4) for _ in range(m)]
    
    # Precompute edge indices
    edge_u = [0] * m
    edge_v = [0] * m
    pair_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_u[idx] = i
            edge_v[idx] = j
            pair_to_idx[(i, j)] = idx
            pair_to_idx[(j, i)] = idx
            idx += 1
            
    # Masks: masks[v][c] is an int bitmask
    masks = [[0]*5 for _ in range(n)]
    
    def build_masks():
        for v in range(n):
            for c in range(5):
                masks[v][c] = 0
        for k in range(m):
            u = edge_u[k]
            v = edge_v[k]
            c = colors[k]
            masks[u][c] |= (1 << v)
            masks[v][c] |= (1 << u)
            
    build_masks()
    
    # Calculate initial violations
    def count_violations():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = bin(masks[u][c] & masks[v][c]).count('1')
                    if common >= 2:
                        total += common - 1
        return total
        
    current_violations = count_violations()
    best_violations = current_violations
    best_colors = colors[:]
    
    if current_violations == 0:
        return True, colors
        
    T = 1.0
    alpha = 0.9999
    min_T = 0.01
    
    start_time = time.time()
    
    # Helper to get cost of pair (u,v) in color c
    def pair_cost(u, v, c):
        if u == v: return 0
        common = bin(masks[u][c] & masks[v][c]).count('1')
        return max(0, common - 1)
        
    while time.time() < deadline:
        # Pick random edge
        k = random.randint(0, m - 1)
        u = edge_u[k]
        v = edge_v[k]
        old_c = colors[k]
        new_c = random.randint(0, 4)
        
        if new_c == old_c:
            continue
            
        # Calculate Delta
        # Pairs affected: (u, w) and (v, w) for all w != u, v. Plus (u, v).
        # We need to sum costs for these pairs before and after change.
        
        # To avoid double counting (u,v) and (v,u), we just iterate w.
        # Affected pairs in old_c:
        # (u, w) for w != u, v. (v, w) for w != u, v. (u, v).
        # Affected pairs in new_c:
        # (u, w) for w != u, v. (v, w) for w != u, v. (u, v).
        
        delta = 0
        
        # Sum old costs for affected pairs in old_c
        old_cost = 0
        for w in range(n):
            if w == u or w == v: continue
            # Pair (u, w)
            if u < w:
                old_cost += pair_cost(u, w, old_c)
            else:
                old_cost += pair_cost(w, u, old_c)
            # Pair (v, w)
            if v < w:
                old_cost += pair_cost(v, w, old_c)
            else:
                old_cost += pair_cost(w, v, old_c)
        # Pair (u, v)
        if u < v:
            old_cost += pair_cost(u, v, old_c)
        else:
            old_cost += pair_cost(v, u, old_c)
            
        # Sum old costs for affected pairs in new_c
        new_cost_old_masks = 0
        for w in range(n):
            if w == u or w == v: continue
            if u < w:
                new_cost_old_masks += pair_cost(u, w, new_c)
            else:
                new_cost_old_masks += pair_cost(w, u, new_c)
            if v < w:
                new_cost_old_masks += pair_cost(v, w, new_c)
            else:
                new_cost_old_masks += pair_cost(w, v, new_c)
        if u < v:
            new_cost_old_masks += pair_cost(u, v, new_c)
        else:
            new_cost_old_masks += pair_cost(v, u, new_c)
            
        # Apply change to masks
        masks[u][old_c] &= ~(1 << v)
        masks[v][old_c] &= ~(1 << u)
        masks[u][new_c] |= (1 << v)
        masks[v][new_c] |= (1 << u)
        colors[k] = new_c
        
        # Sum new costs for affected pairs in new_c
        new_cost_new_masks = 0
        for w in range(n):
            if w == u or w == v: continue
            if u < w:
                new_cost_new_masks += pair_cost(u, w, new_c)
            else:
                new_cost_new_masks += pair_cost(w, u, new_c)
            if v < w:
                new_cost_new_masks += pair_cost(v, w, new_c)
            else:
                new_cost_new_masks += pair_cost(w, v, new_c)
        if u < v:
            new_cost_new_masks += pair_cost(u, v, new_c)
        else:
            new_cost_new_masks += pair_cost(v, u, new_c)
            
        # Sum new costs for affected pairs in old_c (now empty for u,v)
        new_cost_old_masks_after = 0
        for w in range(n):
            if w == u or w == v: continue
            if u < w:
                new_cost_old_masks_after += pair_cost(u, w, old_c)
            else:
                new_cost_old_masks_after += pair_cost(w, u, old_c)
            if v < w:
                new_cost_old_masks_after += pair_cost(v, w, old_c)
            else:
                new_cost_old_masks_after += pair_cost(w, v, old_c)
        if u < v:
            new_cost_old_masks_after += pair_cost(u, v, old_c)
        else:
            new_cost_old_masks_after += pair_cost(v, u, old_c)
            
        # Total Delta
        # Before: old_cost (in old_c) + new_cost_old_masks (in new_c)
        # After: new_cost_old_masks_after (in old_c) + new_cost_new_masks (in new_c)
        delta = (new_cost_old_masks_after + new_cost_new_masks) - (old_cost + new_cost_old_masks)
        
        current_violations += delta
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return True, best_colors
        else:
            # Revert
            colors[k] = old_c
            masks[u][new_c] &= ~(1 << v)
            masks[v][new_c] &= ~(1 << u)
            masks[u][old_c] |= (1 << v)
            masks[v][old_c] |= (1 << u)
            current_violations -= delta
            
        T *= alpha
        if T < min_T:
            T = min_T
            
    if best_violations == 0:
        return True, best_colors
    return False, best_colors

# Re-implement construct to use the fast search correctly
def construct():
    deadline = time.time() + 115
    
    best_n = 11
    # Seed colors for n=11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to find better n
    # We prioritize 26 and 27.
    targets = [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]
    
    for n in targets:
        if time.time() > deadline - 5:
            break
            
        # Time budget per n
        # 27: 20s, 26: 30s, others: 5s
        if n == 27:
            t_limit = 20
        elif n == 26:
            t_limit = 40
        else:
            t_limit = 5
            
        actual_limit = min(deadline - time.time() - 2, t_limit)
        if actual_limit < 1:
            continue
            
        success, colors = _fast_local_search(n, time.time() + actual_limit)
        
        if success:
            best_n = n
            best_colors = colors
            # If we found 27, we are done.
            # If we found 26, we are done (27 failed).
            # If we found 25, we continue? No, 25 < 26.
            # So we can break if we find any n >= 26?
            # Actually, if we find 26, we should return it.
            # We don't need to try 25.
            break
            
    return {'n': best_n, 'colors': best_colors}
