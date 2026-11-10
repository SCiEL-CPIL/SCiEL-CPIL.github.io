import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 12
    m = 40
    max_mask = (1 << n) - 1
    num_masks = 1 << n  # 4096
    
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b
    
    def calc_score(masks):
        score = 0
        for i in range(m):
            for j in range(i + 1, m):
                if is_comparable(masks[i], masks[j]):
                    score += 1
        return score
    
    # Precompute comparability counts for each mask against all others
    # comp_count[a] = number of masks b (b != a) such that a and b are comparable
    # This helps with Lagrangian relaxation
    # For speed, precompute subset/superset relationships
    # A mask a is comparable with b iff a⊆b or b⊆a
    # Number of subsets of a = 2^(popcount(a))
    # Number of supersets of a = 2^(n - popcount(a))
    # Comparable count for mask a (excluding itself) = 2^pop(a) + 2^(n-pop(a)) - 1 - 1
    # Wait, that's not quite right because we need distinct masks and the relation is symmetric
    
    # Actually for the Lagrangian approach, let's think differently:
    # We want to select m masks to maximize sum of comparable pairs
    # This is equivalent to: for each selected mask, count its comparable partners among selected
    
    # Precompute popcount for all masks
    popcount = [0] * num_masks
    for i in range(num_masks):
        popcount[i] = bin(i).count('1')
    
    # Precompute for each mask, the list of masks it's comparable with
    # This is expensive (4096 * up to 4096), but we can be smarter
    # For mask a with popcount k:
    #   Subsets of a: 2^k masks (including a itself)
    #   Supersets of a: 2^(n-k) masks (including a itself)
    #   Total comparable (including a): 2^k + 2^(n-k) - 1
    #   Comparable excluding a: 2^k + 2^(n-k) - 2
    
    # For the Lagrangian relaxation approach:
    # Objective: maximize sum_{i<j, i,j in S} [comparable(i,j)] subject to |S| = m
    # Lagrangian: maximize sum_{i<j, i,j in S} [comparable(i,j)] - lambda * (|S| - m)
    # This doesn't decompose trivially, but we can use it to guide search
    
    # Alternative: use a greedy + local search hybrid with better initialization
    
    # Phase 1: Smart initialization using chain-based structure
    # A chain of length 13 (all subsets of a fixed set of 12 elements in increasing size)
    # gives C(13,2) = 78 comparable pairs from just the chain
    # But we need 40 sets total
    
    # Better: use multiple chains or a structured approach
    # Let's try: take all sets of size 0, 1, 2, ..., 12 that form a "thick" chain
    
    # Strategy: Start with a full chain (13 sets), then add sets that maximize
    # comparability with existing sets
    
    masks = []
    used = set()
    
    # Build a base chain: empty set, {0}, {0,1}, ..., {0,1,2,...,11}
    cur = 0
    for i in range(n + 1):
        masks.append(cur)
        used.add(cur)
        if i < n:
            cur |= (1 << i)
    
    # Now greedily add masks that maximize comparable pairs with existing
    # For each candidate, score = number of existing masks it's comparable with
    # To speed this up, precompute for each mask its comparable count with current set
    
    # Precompute a lookup: for each mask, which other masks is it comparable with?
    # This is 4096 * (avg comparable) which could be large
    # Instead, compute on the fly but cache
    
    # Greedy fill
    all_candidates = list(range(num_masks))
    random.shuffle(all_candidates)
    
    while len(masks) < m:
        if time.time() > deadline - 20:
            break
        best_mask = -1
        best_score = -1
        # Sample a subset of candidates for speed
        sample_size = min(500, len(all_candidates))
        sample = all_candidates[:sample_size]
        for cand in sample:
            if cand in used:
                continue
            score = 0
            for existing in masks:
                if is_comparable(cand, existing):
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = cand
        if best_mask == -1:
            # Find any unused mask
            for cand in all_candidates:
                if cand not in used:
                    best_mask = cand
                    break
        if best_mask == -1:
            break
        masks.append(best_mask)
        used.add(best_mask)
    
    # Ensure we have exactly m masks
    if len(masks) < m:
        for cand in range(num_masks):
            if len(masks) >= m:
                break
            if cand not in used:
                masks.append(cand)
                used.add(cand)
    
    current_score = calc_score(masks)
    best_score = current_score
    best_masks = masks[:]
    
    # Phase 2: Simulated Annealing with improved move generation
    random.seed(42)
    T = 15.0
    cooling = 0.9997
    min_T = 0.05
    
    # Precompute comparability for current set for fast delta calculation
    # For each mask in the set, precompute its comparability with all other masks in the set
    # Store as a bitmask or list
    
    def compute_comparability_vector(mask, other_masks):
        """Return list of 0/1 indicating comparability with each other_mask"""
        return [1 if is_comparable(mask, om) else 0 for om in other_masks]
    
    while time.time() < deadline - 5.0 and T > min_T:
        # Pick random index
        i = random.randint(0, m - 1)
        old_mask = masks[i]
        
        # Generate candidate with bias toward structured moves
        r = random.random()
        if r < 0.5:
            # Small perturbation: flip 1-2 bits
            new_mask = old_mask
            for _ in range(random.randint(1, 2)):
                bit = random.randint(0, n - 1)
                new_mask ^= (1 << bit)
        elif r < 0.8:
            # Move along a chain: add or remove a random element
            if old_mask == 0:
                bit = random.randint(0, n - 1)
                new_mask = old_mask | (1 << bit)
            elif old_mask == max_mask:
                bit = random.randint(0, n - 1)
                new_mask = old_mask & ~(1 << bit)
            else:
                # Try to move toward a denser or sparser set
                if random.random() < 0.5:
                    # Add an element not in old_mask
                    available = [b for b in range(n) if not (old_mask & (1 << b))]
                    if available:
                        bit = random.choice(available)
                        new_mask = old_mask | (1 << bit)
                    else:
                        new_mask = old_mask
                else:
                    # Remove an element from old_mask
                    available = [b for b in range(n) if (old_mask & (1 << b))]
                    if available:
                        bit = random.choice(available)
                        new_mask = old_mask & ~(1 << bit)
                    else:
                        new_mask = old_mask
        else:
            # Larger jump: pick a mask near an existing one
            base = random.choice(masks)
            new_mask = base
            for _ in range(random.randint(2, 4)):
                bit = random.randint(0, n - 1)
                new_mask ^= (1 << bit)
        
        if new_mask == old_mask or new_mask in used:
            T *= cooling
            continue
            
        # Calculate Delta efficiently
        delta = 0
        for j in range(m):
            if j == i:
                continue
            m_j = masks[j]
            old_c = 1 if is_comparable(old_mask, m_j) else 0
            new_c = 1 if is_comparable(new_mask, m_j) else 0
            delta += new_c - old_c
            
        # Accept?
        if delta > 0 or (delta > -50 and random.random() < math.exp(delta / T)):
            used.remove(old_mask)
            used.add(new_mask)
            masks[i] = new_mask
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_masks = masks[:]
            
            T *= cooling
    
    # Phase 3: Exact local search polish
    polish_deadline = deadline - 1.0
    polish_masks = best_masks[:]
    polish_score = best_score
    polish_used = set(polish_masks)
    
    improved = True
    polish_iter = 0
    while improved and time.time() < polish_deadline:
        improved = False
        polish_iter += 1
        for i in range(m):
            if time.time() > polish_deadline:
                break
            old_mask = polish_masks[i]
            best_new = old_mask
            best_delta = 0
            # Try all single-bit flips and some 2-bit flips
            candidates = []
            for bit in range(n):
                candidates.append(old_mask ^ (1 << bit))
            # Add some 2-bit flips
            for b1 in range(n):
                for b2 in range(b1 + 1, n):
                    candidates.append(old_mask ^ (1 << b1) ^ (1 << b2))
            
            for new_mask in candidates:
                if new_mask in polish_used:
                    continue
                delta = 0
                for j in range(m):
                    if j == i:
                        continue
                    m_j = polish_masks[j]
                    old_c = 1 if is_comparable(old_mask, m_j) else 0
                    new_c = 1 if is_comparable(new_mask, m_j) else 0
                    delta += new_c - old_c
                if delta > best_delta:
                    best_delta = delta
                    best_new = new_mask
            if best_delta > 0:
                polish_used.remove(old_mask)
                polish_used.add(best_new)
                polish_masks[i] = best_new
                polish_score += best_delta
                improved = True
                if polish_score > best_score:
                    best_score = polish_score
                    best_masks = polish_masks[:]
    
    # Final conversion
    result = []
    for mask in best_masks:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    # Ensure exactly 40 distinct lists with proper types
    if len(result) != m:
        seen = set()
        final = []
        for r in result:
            t = tuple(r)
            if t not in seen:
                seen.add(t)
                final.append(list(r))
        while len(final) < m:
            found = False
            for mask in range(num_masks):
                t = tuple([b for b in range(n) if (mask >> b) & 1])
                if t not in seen:
                    seen.add(t)
                    final.append([b for b in range(n) if (mask >> b) & 1])
                    found = True
                    break
            if not found:
                break
        result = final[:m]
    
    # Final validation and type enforcement
    final_result = []
    for r in result:
        # Ensure all elements are Python int
        clean_r = [int(x) for x in r]
        # Ensure distinct
        if len(set(clean_r)) == len(clean_r):
            final_result.append(clean_r)
    
    # If we lost any due to validation, refill
    if len(final_result) < m:
        seen = set(tuple(r) for r in final_result)
        for mask in range(num_masks):
            if len(final_result) >= m:
                break
            t = tuple([b for b in range(n) if (mask >> b) & 1])
            if t not in seen:
                seen.add(t)
                final_result.append([b for b in range(n) if (mask >> b) & 1])
    
    return final_result[:m]
