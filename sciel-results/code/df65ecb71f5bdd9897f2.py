import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit (evaluator kills at 90s)
    
    n = 12
    m = 40
    total_masks = 1 << n  # 4096 possible subsets
    
    # Precompute popcount
    popcount = [0] * total_masks
    for mask in range(1, total_masks):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)
    
    # Precompute comparability matrix for all 4096 masks
    # comp[i][j] = True if masks i and j are comparable (one is strict subset of other)
    # Use a list of bytearrays for memory efficiency
    comp = [bytearray(total_masks) for _ in range(total_masks)]
    for i in range(total_masks):
        mi = i
        for j in range(i + 1, total_masks):
            mj = j
            if (mi & mj) == mi or (mi & mj) == mj:
                if mi != mj:
                    comp[i][j] = 1
                    comp[j][i] = 1
    
    def count_comparable_fast(family):
        count = 0
        k = len(family)
        for i in range(k):
            a = family[i]
            row = comp[a]
            for j in range(i + 1, k):
                if row[family[j]]:
                    count += 1
        return count
    
    # Precompute for each mask, a bitset of which masks it's comparable with
    # comparable_bits[m]: 4096-bit integer, bit i set if mask m is comparable with mask i
    comparable_bits = [0] * total_masks
    for i in range(total_masks):
        bits = 0
        for j in range(total_masks):
            if i != j and comp[i][j]:
                bits |= (1 << j)
        comparable_bits[i] = bits
    
    def popcount_int(x):
        return bin(x).count('1')
    
    # Build initial solution: greedy chain-based
    random.seed(42)
    
    # Start with a full chain of all sizes
    chain = [(1 << size) - 1 for size in range(13)]
    family = list(chain)
    family_set = set(family)
    family_bits = 0
    for mask in family:
        family_bits |= (1 << mask)
    
    # Greedily add 27 more masks
    candidates = list(range(total_masks))
    random.shuffle(candidates)
    candidates = [c for c in candidates if c not in family_set]
    
    for _ in range(m - len(family)):
        best_mask = None
        best_gain = -1
        # Search a subset of candidates for speed
        for c in candidates[:1000]:
            gain = popcount_int(comparable_bits[c] & family_bits)
            if gain > best_gain:
                best_gain = gain
                best_mask = c
        if best_mask is not None:
            family.append(best_mask)
            family_set.add(best_mask)
            family_bits |= (1 << best_mask)
            candidates.remove(best_mask)
    
    best_score = count_comparable_fast(family)
    best_family = list(family)
    
    # Local search: single-move
    def try_single_replacement(family, family_bits, family_set):
        """Try to improve by replacing one mask. Returns (new_family, new_bits, new_set, improved)"""
        k = len(family)
        for i in range(k):
            old_mask = family[i]
            old_degree = popcount_int(comparable_bits[old_mask] & (family_bits ^ (1 << old_mask)))
            
            best_new_mask = None
            best_new_degree = -1
            
            # Try all masks not in family
            for c in range(total_masks):
                if c in family_set:
                    continue
                new_degree = popcount_int(comparable_bits[c] & (family_bits ^ (1 << old_mask)))
                if new_degree > best_new_degree:
                    best_new_degree = new_degree
                    best_new_mask = c
            
            if best_new_mask is not None and best_new_degree > old_degree:
                new_family = list(family)
                new_family[i] = best_new_mask
                new_set = set(family_set)
                new_set.remove(old_mask)
                new_set.add(best_new_mask)
                new_bits = family_bits ^ (1 << old_mask) | (1 << best_new_mask)
                return new_family, new_bits, new_set, True
        
        return family, family_bits, family_set, False
    
    # Multi-start local search with simulated annealing-like perturbation
    restarts = 0
    no_improve_count = 0
    max_no_improve = 15
    
    while time.time() < deadline:
        restarts += 1
        no_improve_count = 0
        
        # Perturb the best solution
        if restarts == 1:
            family = list(best_family)
            family_set = set(family)
            family_bits = 0
            for mask in family:
                family_bits |= (1 << mask)
        else:
            family = list(best_family)
            family_set = set(family)
            family_bits = 0
            for mask in family:
                family_bits |= (1 << mask)
            
            # Randomly replace 2-6 masks with increasing perturbation
            num_replace = random.randint(2, min(6, restarts))
            indices = random.sample(range(m), num_replace)
            for idx in indices:
                old_mask = family[idx]
                family_set.remove(old_mask)
                family_bits ^= (1 << old_mask)
                # Pick a random new mask
                while True:
                    new_mask = random.randint(0, total_masks - 1)
                    if new_mask not in family_set:
                        break
                family[idx] = new_mask
                family_set.add(new_mask)
                family_bits |= (1 << new_mask)
        
        # Local search: repeatedly try single replacements until no improvement
        while time.time() < deadline - 1.0:
            new_family, new_bits, new_set, improved = try_single_replacement(family, family_bits, family_set)
            if not improved:
                break
            family = new_family
            family_bits = new_bits
            family_set = new_set
            no_improve_count = 0
        
        score = count_comparable_fast(family)
        if score > best_score:
            best_score = score
            best_family = list(family)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve and restarts > 5:
                # Try a more aggressive perturbation
                if restarts > 10:
                    break
    
    # Final: try more aggressive local search on best_family with 2-move exchanges
    family = list(best_family)
    family_set = set(family)
    family_bits = 0
    for mask in family:
        family_bits |= (1 << mask)
    
    # Try 2-move exchanges for final polish
    while time.time() < deadline - 0.5:
        improved = False
        k = len(family)
        for i in range(k):
            if time.time() > deadline - 0.5:
                break
            old_i = family[i]
            old_deg_i = popcount_int(comparable_bits[old_i] & (family_bits ^ (1 << old_i)))
            
            best_new_i = None
            best_new_deg_i = -1
            
            # Try all masks not in family
            for c in range(total_masks):
                if c in family_set:
                    continue
                new_deg_i = popcount_int(comparable_bits[c] & (family_bits ^ (1 << old_i)))
                if new_deg_i > best_new_deg_i:
                    best_new_deg_i = new_deg_i
                    best_new_i = c
            
            if best_new_i is not None and best_new_deg_i > old_deg_i:
                # Apply the move
                family_set.remove(old_i)
                family_bits ^= (1 << old_i)
                family[i] = best_new_i
                family_set.add(best_new_i)
                family_bits |= (1 << best_new_i)
                improved = True
        
        if not improved:
            break
    
    # Final single-move local search
    while time.time() < deadline - 0.1:
        new_family, new_bits, new_set, improved = try_single_replacement(family, family_bits, family_set)
        if not improved:
            break
        family = new_family
        family_bits = new_bits
        family_set = new_set
    
    score = count_comparable_fast(family)
    if score > best_score:
        best_score = score
        best_family = list(family)
    
    # Convert masks to lists of integers
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    # Ensure types are correct: lists of ints
    for i in range(len(result)):
        result[i] = [int(x) for x in result[i]]
    
    return result
