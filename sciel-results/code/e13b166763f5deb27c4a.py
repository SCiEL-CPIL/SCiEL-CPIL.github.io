import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    n = 12
    m = 40
    deadline = time.time() + 85  # Safety margin (90s limit)
    
    total_masks = 1 << n  # 4096 possible masks
    
    # Precompute subset relationship for all pairs of masks
    # is_sub[a][b] = 1 if mask a is a subset of mask b
    # We'll use a dictionary or array for fast lookup
    # Actually, let's compute on the fly but cache results
    
    # Helper to check if a is subset of b
    def is_subset(a, b):
        return (a & b) == a
    
    # Helper to calculate score of a list of masks
    def calc_score(masks):
        score = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if is_subset(masks[i], masks[j]) or is_subset(masks[j], masks[i]):
                    score += 1
        return score
    
    # Greedy initialization: build a set that maximizes chain-like structure
    def greedy_init():
        # Start with some masks that form a chain
        # Then add more masks that are comparable to many existing ones
        masks = []
        used = set()
        
        # First, add a long chain: {}, {0}, {0,1}, {0,1,2}, ...
        # This gives us a base of comparable pairs
        for k in range(1, n + 1):
            mask = (1 << k) - 1  # {0, 1, ..., k-1}
            if mask not in used:
                masks.append(mask)
                used.add(mask)
        
        # Add the complement chain: {11}, {10,11}, {9,10,11}, ...
        for k in range(1, n + 1):
            mask = ((1 << k) - 1) << (n - k)  # {n-k, ..., n-1}
            if mask not in used:
                masks.append(mask)
                used.add(mask)
        
        # Now greedily add remaining masks
        # Score for adding a new mask: number of existing masks it's comparable with
        while len(masks) < m:
            best_mask = -1
            best_score = -1
            # Try random candidates first, then systematic if needed
            candidates = []
            for _ in range(200):
                cand = random.getrandbits(n)
                if cand not in used:
                    candidates.append(cand)
            
            if not candidates:
                # Systematic search
                for cand in range(total_masks):
                    if cand not in used:
                        candidates.append(cand)
                if not candidates:
                    break
            
            for cand in candidates:
                s = 0
                for existing in masks:
                    if is_subset(cand, existing) or is_subset(existing, cand):
                        s += 1
                if s > best_score:
                    best_score = s
                    best_mask = cand
            
            if best_mask == -1:
                break
            
            masks.append(best_mask)
            used.add(best_mask)
        
        return masks, used
    
    # Initialize with greedy construction
    current_masks, used = greedy_init()
    
    # If we don't have enough, fill randomly
    while len(current_masks) < m:
        mask = random.getrandbits(n)
        if mask not in used:
            used.add(mask)
            current_masks.append(mask)
    
    # Ensure exactly m masks
    current_masks = current_masks[:m]
    used = set(current_masks)
    
    current_score = calc_score(current_masks)
    best_masks = current_masks[:]
    best_score = current_score
    
    # Local Search with Annealing
    T = 2.0
    no_improve_count = 0
    max_no_improve = 2000
    T_decay = 0.9995
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        i = random.randint(0, m - 1)
        # Pick a new mask not in the set
        new_mask = random.getrandbits(n)
        if new_mask in used:
            continue
        
        # Calculate delta
        old_mask = current_masks[i]
        delta = 0
        for j in range(m):
            if j == i:
                continue
            other = current_masks[j]
            # Old contribution
            if is_subset(old_mask, other) or is_subset(other, old_mask):
                delta -= 1
            # New contribution
            if is_subset(new_mask, other) or is_subset(other, new_mask):
                delta += 1
        
        # Accept if better, or with probability exp(delta/T) if worse
        if delta > 0:
            accept = True
        else:
            if T > 0.01:
                accept = random.random() < math.exp(delta / T)
            else:
                accept = False
        
        if accept:
            used.remove(old_mask)
            current_masks[i] = new_mask
            used.add(new_mask)
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_masks = current_masks[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Decrease temperature
            T *= T_decay
            
            # Check time
            if time.time() > deadline - 5:
                break
    
    # Final polish: try all possible single replacements for each position
    # This is O(m * total_masks * m) = 40 * 4096 * 40 ≈ 6.5M operations, should be fast
    for i in range(m):
        old_mask = best_masks[i]
        best_new_mask = old_mask
        best_delta = 0
        
        for new_mask in range(total_masks):
            if new_mask == old_mask:
                continue
            if new_mask in set(best_masks):
                continue
            
            # Calculate delta for replacing index i with new_mask
            delta = 0
            for j in range(m):
                if j == i:
                    continue
                other = best_masks[j]
                # Old contribution
                if is_subset(old_mask, other) or is_subset(other, old_mask):
                    delta -= 1
                # New contribution
                if is_subset(new_mask, other) or is_subset(other, new_mask):
                    delta += 1
            
            if delta > best_delta:
                best_delta = delta
                best_new_mask = new_mask
        
        if best_delta > 0:
            best_masks[i] = best_new_mask
    
    # One more polish pass
    for i in range(m):
        old_mask = best_masks[i]
        best_new_mask = old_mask
        best_delta = 0
        
        for new_mask in range(total_masks):
            if new_mask == old_mask:
                continue
            if new_mask in set(best_masks):
                continue
            
            # Calculate delta for replacing index i with new_mask
            delta = 0
            for j in range(m):
                if j == i:
                    continue
                other = best_masks[j]
                # Old contribution
                if is_subset(old_mask, other) or is_subset(other, old_mask):
                    delta -= 1
                # New contribution
                if is_subset(new_mask, other) or is_subset(other, new_mask):
                    delta += 1
            
            if delta > best_delta:
                best_delta = delta
                best_new_mask = new_mask
        
        if best_delta > 0:
            best_masks[i] = best_new_mask
    
    # Convert masks to lists of ints
    result = []
    for mask in best_masks:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
# EVOLVE-BLOCK-END
