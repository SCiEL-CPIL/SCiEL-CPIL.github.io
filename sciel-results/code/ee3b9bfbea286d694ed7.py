"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin for 90s kill
    
    n = 12
    m = 40
    N = 1 << n  # 4096
    
    # Precompute subset counts and subset relations efficiently
    # bit_count[i] = number of 1 bits in i
    bit_count = [0] * N
    for i in range(1, N):
        bit_count[i] = bit_count[i >> 1] + (i & 1)
    
    # For each mask, precompute list of masks that are strict supersets
    # superset_count[i] = number of masks j such that i is strict subset of j
    # We can compute this via DP over subsets
    # Actually, let's think about the score differently.
    
    # Score = number of unordered pairs (a, b) where a is strict subset of b
    # For a family F of m sets, score = sum over all pairs {a,b} in F of [a ⊂ b or b ⊂ a]
    
    # Key insight: if we order sets by size, then a ⊂ b implies |a| < |b|
    # So we only need to check pairs where |a| < |b|
    
    # Let's use a greedy/local search approach:
    # 1. Start with a good initial family
    # 2. Try to improve by swapping one set for another
    
    # Initial family: use a chain-like structure
    # The maximum possible score is C(m, 2) = C(40, 2) = 780 if we had a total order
    # But we can't have a total order of 40 distinct subsets of 12 elements
    # since there are only 4096 subsets total, and chains are limited by n+1 = 13
    
    # Best strategy: use multiple chains or a layered structure
    # Let's try: for each size k from 0 to 12, pick several sets of that size
    # and ensure comparability within and across layers
    
    # Actually, the optimal structure for maximizing comparable pairs is to use
    # as many chains as possible. By Dilworth's theorem, the minimum number of
    # chains needed to cover a poset equals the size of the largest antichain.
    
    # For the Boolean lattice of dimension 12, the largest antichain is C(12, 6) = 924
    # So we need at least 924 chains to cover the whole lattice, but we only need 40 sets.
    
    # A simple good heuristic: pick 40 sets that form as many chains as possible.
    # One approach: pick sets of sizes 0, 1, 2, ..., 12 with overlaps.
    
    # Let's try a constructive approach:
    # Start with the empty set, then add sets that are supersets of existing ones
    # to create chains.
    
    # Better: use a random restart + local search approach
    # Score function: count comparable pairs
    
    def score_family(family_masks):
        """Count number of unordered pairs where one is strict subset of other."""
        count = 0
        m_local = len(family_masks)
        for i in range(m_local):
            for j in range(i + 1, m_local):
                a, b = family_masks[i], family_masks[j]
                # Check if a is strict subset of b or b is strict subset of a
                if a != b:
                    if (a & b) == a:  # a subset of b
                        count += 1
                    elif (a & b) == b:  # b subset of a
                        count += 1
        return count
    
    # Precompute subset check table for speed
    # subset_check[a][b] = True if a is strict subset of b
    # This is 4096 x 4096 = 16M entries, which is ~16MB as a list of bytearrays
    # Let's use a more memory-efficient approach: just use the bitwise check
    
    # Actually, let's precompute for each mask the list of masks that are its strict subsets
    # and strict supersets, but that might be too much memory.
    
    # Let's use a different approach: for scoring, sort by bit count and only check
    # pairs where bit_count[i] < bit_count[j]
    
    def score_family_fast(family_masks):
        """Faster scoring using bit count sorting."""
        # Sort by bit count
        indexed = [(bit_count[x], x) for x in family_masks]
        indexed.sort()
        count = 0
        m_local = len(family_masks)
        for i in range(m_local):
            for j in range(i + 1, m_local):
                if indexed[i][0] >= indexed[j][0]:
                    continue
                a, b = indexed[i][1], indexed[j][1]
                if (a & b) == a:  # a is strict subset of b (since sizes differ)
                    count += 1
        return count
    
    # Initial family: try a few heuristics
    # Heuristic 1: Greedy chain building
    def build_greedy_chain():
        """Build a family by greedily adding sets that maximize comparability."""
        family = [0]  # start with empty set
        used = {0}
        
        # Precompute all masks sorted by bit count
        all_masks_by_size = {}
        for mask in range(N):
            bc = bit_count[mask]
            if bc not in all_masks_by_size:
                all_masks_by_size[bc] = []
            all_masks_by_size[bc].append(mask)
        
        # For each size, we'll pick sets that are comparable to existing family members
        # Try to build chains: pick a base, then extend with supersets
        
        # Alternative: pick random chains
        random.seed(42)
        
        # Try multiple random initializations and keep the best
        best_family = None
        best_score = -1
        
        for attempt in range(50):
            # Build a random family
            family = []
            used = set()
            
            # Pick random chains
            num_chains = random.randint(5, 15)
            for c in range(num_chains):
                # Start from a random small set
                start_size = random.randint(0, 4)
                start_mask = random.choice(all_masks_by_size.get(start_size, [0]))
                if start_mask in used:
                    continue
                family.append(start_mask)
                used.add(start_mask)
                
                # Extend the chain with supersets
                current = start_mask
                while len(family) < m:
                    # Find a superset of current that's not used
                    # A superset must have all bits of current plus at least one more
                    # Enumerate all supersets of current
                    free_bits = [b for b in range(n) if not (current >> b) & 1]
                    if not free_bits:
                        break
                    # Pick a random subset of free bits to add
                    num_add = random.randint(1, min(3, len(free_bits)))
                    add_bits = random.sample(free_bits, num_add)
                    new_mask = current
                    for b in add_bits:
                        new_mask |= (1 << b)
                    if new_mask in used:
                        # Try a different choice
                        if random.random() < 0.5:
                            break
                        continue
                    family.append(new_mask)
                    used.add(new_mask)
                    current = new_mask
                
                if len(family) >= m:
                    break
            
            # Fill remaining with random sets
            while len(family) < m:
                mask = random.randint(0, N - 1)
                if mask not in used:
                    family.append(mask)
                    used.add(mask)
            
            if len(family) != m:
                continue
                
            s = score_family_fast(family)
            if s > best_score:
                best_score = s
                best_family = family[:]
            
            if time.time() > deadline - 5:
                break
        
        if best_family is not None:
            return best_family
        
        # Fallback: first m masks
        return list(range(m))
    
    # Local search: try to improve by swapping one set for another
    def local_search(family, max_iters=1000):
        """Improve family by swapping sets."""
        current_score = score_family_fast(family)
        best_family = family[:]
        best_score = current_score
        
        # Precompute for each set in family, the sets it's comparable with
        # To speed up swap evaluation, we can compute the marginal contribution
        
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            improved = False
            # Try swapping each set with a random candidate
            for i in range(m):
                if time.time() > deadline - 10:
                    break
                
                # Try a few random candidates
                for _ in range(5):
                    new_mask = random.randint(0, N - 1)
                    if new_mask in best_family:
                        continue
                    
                    # Evaluate swap: remove best_family[i], add new_mask
                    # Compute change in score
                    old_set = best_family[i]
                    
                    # Count how many other sets are comparable with old_set
                    old_comparable = 0
                    for j in range(m):
                        if j == i:
                            continue
                        a, b = old_set, best_family[j]
                        if a != b:
                            if (a & b) == a or (a & b) == b:
                                old_comparable += 1
                    
                    # Count how many other sets are comparable with new_mask
                    new_comparable = 0
                    for j in range(m):
                        if j == i:
                            continue
                        a, b = new_mask, best_family[j]
                        if a != b:
                            if (a & b) == a or (a & b) == b:
                                new_comparable += 1
                    
                    delta = new_comparable - old_comparable
                    if delta > 0:
                        best_family[i] = new_mask
                        best_score += delta
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # No improvement found, try a random perturbation
                if random.random() < 0.3:
                    i = random.randint(0, m - 1)
                    new_mask = random.randint(0, N - 1)
                    if new_mask not in best_family:
                        best_family[i] = new_mask
                        best_score = score_family_fast(best_family)
        
        return best_family, best_score
    
    # Main search
    best_family = None
    best_score = -1
    
    # Try multiple initializations
    for init in range(20):
        if time.time() > deadline - 15:
            break
        
        # Build initial family
        if init == 0:
            # Greedy chain
            family = build_greedy_chain()
        else:
            # Random family with structure
            random.seed(100 + init)
            family = []
            used = set()
            
            # Pick some chains
            for _ in range(random.randint(3, 10)):
                start_size = random.randint(0, 5)
                all_of_size = [x for x in range(N) if bit_count[x] == start_size and x not in used]
                if not all_of_size:
                    continue
                current = random.choice(all_of_size)
                family.append(current)
                used.add(current)
                
                while len(family) < m:
                    free_bits = [b for b in range(n) if not (current >> b) & 1]
                    if not free_bits:
                        break
                    num_add = random.randint(1, min(2, len(free_bits)))
                    add_bits = random.sample(free_bits, num_add)
                    new_mask = current
                    for b in add_bits:
                        new_mask |= (1 << b)
                    if new_mask in used:
                        continue
                    family.append(new_mask)
                    used.add(new_mask)
                    current = new_mask
            
            while len(family) < m:
                mask = random.randint(0, N - 1)
                if mask not in used:
                    family.append(mask)
                    used.add(mask)
        
        if len(family) != m:
            continue
        
        # Local search
        family, score = local_search(family, max_iters=500)
        
        if score > best_score:
            best_score = score
            best_family = family[:]
        
        if time.time() > deadline - 10:
            break
    
    # Final result
    if best_family is None:
        # Fallback
        best_family = list(range(m))
    
    # Convert to lists of integers
    result = []
    for mask in best_family:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
# EVOLVE-BLOCK-END
