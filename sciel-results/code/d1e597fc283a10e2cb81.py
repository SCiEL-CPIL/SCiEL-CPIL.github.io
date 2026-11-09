import time
import random
import math

N_VERT = 12
M = 40
TIME_LIMIT = 75  # seconds, leave 15s margin for 90s wall-clock limit

def _count_pairs(masks):
    """Count comparable pairs in a list of integer masks."""
    count = 0
    m = len(masks)
    for i in range(m):
        a = masks[i]
        for j in range(i + 1, m):
            b = masks[j]
            if (a & b) == a or (a & b) == b:
                count += 1
    return count

def _delta_for_swap(masks, i, new_mask):
    """Compute change in comparable-pair count when masks[i] is replaced by new_mask."""
    old = masks[i]
    if new_mask == old:
        return 0
    delta = 0
    for j in range(M):
        if j == i:
            continue
        m = masks[j]
        # Remove contribution of (old, m)
        if (old & m) == old or (old & m) == m:
            delta -= 1
        # Add contribution of (new_mask, m)
        if (new_mask & m) == new_mask or (new_mask & m) == m:
            delta += 1
    return delta

def _greedy_init():
    """Greedy construction: start with empty, add mask that maximizes marginal gain."""
    all_masks = list(range(1 << N_VERT))
    # Sort by popcount to encourage chain-like structure? Or just random order.
    # Let's try: start with empty set, then greedily add.
    # But we need exactly 40 distinct.
    # Better: initialize with a known good structure? 
    # Let's use a simple greedy: pick 40 masks that form a dense comparable structure.
    # Actually, a chain of 40 sets would give 40*39/2 = 780 pairs. Max possible.
    # Can we have a chain of 40 distinct subsets of 12 elements?
    # Yes, e.g., {0}, {0,1}, {0,1,2}, ... but only 13 in a chain if we add one each time.
    # We can have longer chains by using different subsets.
    # Max chain length in B_12 is 13 (empty to full). So a pure chain gives at most 13 sets.
    # For 40 sets, we need multiple chains or a broad structure.
    
    # Let's just do random init for simplicity, then optimize.
    # But random init might be poor. Let's try a structured init:
    # Use the 40 masks with the smallest popcounts? Or a mix?
    # Let's use random for now, but with a seed.
    random.seed(42)
    return random.sample(all_masks, M)

def _local_search(masks, deadline):
    """2-opt local search: at each step, try all possible replacements for each position."""
    current = list(masks)
    current_score = _count_pairs(current)
    best = list(current)
    best_score = current_score
    
    while time.time() < deadline:
        improved = False
        # For each position, find the best replacement
        # To speed up, we can precompute the set of masks in current
        current_set = set(current)
        
        for i in range(M):
            old = current[i]
            best_delta = 0
            best_new = None
            
            # Iterate over all possible masks
            # 4096 masks, for each of 40 positions = 163840 evaluations per pass
            # Each eval is O(40) = 6553600 ops per pass. 
            # In Python, this might be too slow. 
            # Let's estimate: 6.5M simple ops per pass. Python does ~10M-100M simple ops/sec? 
            # Maybe 0.1-0.6s per pass. We can afford ~100-700 passes in 75s.
            
            for new_mask in range(1 << N_VERT):
                if new_mask in current_set:
                    continue
                d = _delta_for_swap(current, i, new_mask)
                if d > best_delta:
                    best_delta = d
                    best_new = new_mask
            
            if best_new is not None and best_delta > 0:
                current[i] = best_new
                current_set.remove(old)
                current_set.add(best_new)
                current_score += best_delta
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
        
        if not improved:
            break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Initialize
    masks = _greedy_init()
    
    # Local search
    best_masks, best_score = _local_search(masks, deadline)
    
    # Convert to list of lists of ints
    result = []
    for mask in best_masks:
        result.append([b for b in range(N_VERT) if (mask >> b) & 1])
    
    return result
