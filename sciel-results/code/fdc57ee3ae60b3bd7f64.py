import time
import random
import math

def _count_comparable_pairs(family):
    """Count unordered pairs (i,j) where family[i] is a strict subset of family[j] or vice versa."""
    n = len(family)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = family[i], family[j]
            if (a & b) == a:  # a subset of b
                if a != b:
                    count += 1
            elif (a & b) == b:  # b subset of a
                if a != b:
                    count += 1
    return count

def _chain_based_initial():
    """
    Generate 40 distinct subsets of {0..11} that form a near-chain structure.
    Strategy: Build a long chain of subsets, then add extra sets that are comparable
    with many chain members. A pure chain of 40 would give C(40,2)=780 pairs.
    We need 40 distinct subsets from a 12-element set (max 4096 possible).
    
    Approach: 
    1. Generate a Hamiltonian-like path through the subset lattice.
    2. Use a greedy approach: start with empty set, repeatedly add/remove one element
       to create a long chain, then fill remaining slots with sets comparable to many.
    """
    N = 12
    M = 40
    
    # Generate a long chain by walking through the subset lattice
    # Start with empty set, greedily extend by adding one bit, then remove, etc.
    chain = [0]
    chain_set = {0}
    current = 0
    
    # Try to build a long chain by adding bits in a specific order
    # Then remove them, creating a zigzag path
    # Order of bits to add: 0,1,2,...,11
    # Then remove in reverse or some other order
    
    # Strategy: Build a chain by adding bits in order 0,1,...,11
    # That gives 13 sets: {}, {0}, {0,1}, ..., {0,1,...,11}
    # Then we need 27 more. We can create branches.
    
    # Better approach: Use a space-filling curve on the hypercube
    # Gray code gives a Hamiltonian path on the 12-cube (4096 vertices)
    # Consecutive Gray code values differ by 1 bit, so they're comparable
    # if one has exactly 1 more bit than the other.
    
    # Actually, in Gray code, consecutive elements differ by exactly one bit flip.
    # If we flip a 0 to 1, the new set is a superset. If 1 to 0, it's a subset.
    # So consecutive Gray code elements are always comparable!
    # A Gray code of length 40 gives 39 comparable pairs from adjacency.
    # But non-adjacent elements may also be comparable.
    
    # Let's use Gray code to generate 40 consecutive values, which gives us
    # a path where each consecutive pair is comparable. Then we get at least 39 pairs
    # from the path, plus additional pairs from non-adjacent comparabilities.
    
    def gray_code(n):
        """Generate first n Gray code values."""
        for i in range(n):
            yield i ^ (i >> 1)
    
    # Generate 40 consecutive Gray code values
    # These form a path where consecutive elements are comparable
    gray_vals = list(gray_code(M))
    
    # Convert to bitmasks (Gray code value directly is the bitmask)
    # Wait, Gray code value g = i ^ (i>>1) is the binary representation.
    # The actual subset is the set of bits in g.
    # Consecutive Gray codes differ by one bit, so they're comparable.
    
    # But we want to maximize total comparable pairs, not just path edges.
    # Let's try a different approach: build a chain explicitly.
    
    # Chain approach: 
    # Start with {}, add bit 0, add bit 1, ..., add bit 11: that's 13 sets.
    # Then remove bit 0, add bit... no, that breaks the chain.
    
    # Better: Use a "snake" through the lattice.
    # Level 0: {} (1 set)
    # Level 1: {0}, {1}, ... (12 sets) - but we can only pick comparable ones
    # 
    # Actually, the maximum size of a chain in the 12-cube is 13 (empty to full).
    # So we can't have 40 sets in a single chain.
    # 
    # We need a union of chains. The optimal structure is likely:
    # - Several chains of various lengths
    # - Cross-comparabilities between chains
    #
    # Let's use a greedy approach with good initialization:
    # 1. Start with a long chain: {}, {0}, {0,1}, ..., {0,...,11} (13 sets)
    # 2. Add sets that are comparable with many existing sets
    
    # Build initial chain
    F = [0]
    F_set = {0}
    for i in range(N):
        new_mask = F[-1] | (1 << i)
        F.append(new_mask)
        F_set.add(new_mask)
    # Now F has 13 sets forming a chain: 0 ⊂ {0} ⊂ {0,1} ⊂ ... ⊂ {0..11}
    # That's C(13,2) = 78 comparable pairs from this chain alone.
    
    # Now add 27 more sets greedily
    all_masks = list(range(1 << N))
    random.shuffle(all_masks)
    
    for _ in range(M - len(F)):
        best_mask = -1
        best_score = -1
        for mask in all_masks:
            if mask in F_set:
                continue
            score = 0
            for m in F:
                if m != mask:
                    if (m & mask) == m or (m & mask) == mask:
                        score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
                if best_score == len(F):
                    break
        if best_mask != -1:
            F.append(best_mask)
            F_set.add(best_mask)
        else:
            for mask in all_masks:
                if mask not in F_set:
                    F.append(mask)
                    F_set.add(mask)
                    break
    
    return F

def _local_search(F, time_budget):
    """Swap-based local search with simulated annealing, optimized with precomputed comparability."""
    N = 12
    M = len(F)
    deadline = time.time() + time_budget
    F = F[:]
    F_set = set(F)
    
    # Precompute subset/superset relationships for all 4096 masks
    # For each mask, precompute which other masks are subsets or supersets
    # This is expensive to precompute fully, so we'll use a different approach.
    
    # Instead, precompute popcount and use it for quick filtering
    popcount = [bin(m).count('1') for m in range(1 << N)]
    
    def is_comparable(a, b):
        if a == b:
            return False
        # a subset of b iff (a & b) == a
        # b subset of a iff (a & b) == b
        return (a & b) == a or (a & b) == b
    
    def calc_score(fam):
        s = 0
        for i in range(M):
            for j in range(i + 1, M):
                a, b = fam[i], fam[j]
                if a != b and ((a & b) == a or (a & b) == b):
                    s += 1
        return s
    
    current_score = calc_score(F)
    best_score = current_score
    best_F = F[:]
    
    T = 5.0
    T_min = 0.01
    alpha = 0.9997
    
    # Precompute for each mask in F, the list of masks it's comparable with
    # This helps with delta computation
    
    while time.time() < deadline and T > T_min:
        idx = random.randint(0, M - 1)
        old_m = F[idx]
        
        # Generate new mask by flipping 1-3 bits
        r = random.random()
        if r < 0.6:
            flips = 1
        elif r < 0.9:
            flips = 2
        else:
            flips = 3
        
        new_m = old_m
        for _ in range(flips):
            bit = random.randint(0, N - 1)
            new_m ^= (1 << bit)
        
        if new_m in F_set:
            T *= alpha
            continue
        
        # Compute delta: change in comparable pairs
        # Remove contributions of old_m with all other masks
        # Add contributions of new_m with all other masks
        delta = 0
        for i, m in enumerate(F):
            if i == idx:
                continue
            # Old contribution
            if old_m != m and ((old_m & m) == old_m or (old_m & m) == m):
                delta -= 1
            # New contribution
            if new_m != m and ((new_m & m) == new_m or (new_m & m) == m):
                delta += 1
        
        if delta > 0 or random.random() < math.exp(delta / T):
            F_set.remove(old_m)
            F_set.add(new_m)
            F[idx] = new_m
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_F = F[:]
        
        T *= alpha
    
    return best_F, best_score

def _branch_and_bound_polish(F, time_budget):
    """
    Branch and bound to find better local assignments.
    For each position, try all possible masks and pick the best one.
    Use bounding to prune.
    """
    N = 12
    M = len(F)
    deadline = time.time() + time_budget
    F = F[:]
    F_set = set(F)
    
    def calc_score(fam):
        s = 0
        for i in range(M):
            for j in range(i + 1, M):
                a, b = fam[i], fam[j]
                if a != b and ((a & b) == a or (a & b) == b):
                    s += 1
        return s
    
    current_score = calc_score(F)
    best_score = current_score
    best_F = F[:]
    
    # For each position, find the best mask to put there
    # This is a greedy coordinate descent with bounding
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(M):
            old_m = F[idx]
            F_set.remove(old_m)
            
            best_mask = old_m
            best_delta = 0
            
            # Try all possible masks (4096)
            # But we can bound: the maximum possible improvement is M-1
            for mask in range(1 << N):
                if mask in F_set:
                    continue
                
                # Compute delta for this mask
                delta = 0
                for i, m in enumerate(F):
                    if i == idx:
                        continue
                    # Old contribution
                    if old_m != m and ((old_m & m) == old_m or (old_m & m) == m):
                        delta -= 1
                    # New contribution
                    if mask != m and ((mask & m) == mask or (mask & m) == m):
                        delta += 1
                
                if delta > best_delta:
                    best_delta = delta
                    best_mask = mask
                    if best_delta == M - 1:  # Can't do better
                        break
            
            if best_mask != old_m and best_delta > 0:
                F[idx] = best_mask
                F_set.add(best_mask)
                current_score += best_delta
                if current_score > best_score:
                    best_score = current_score
                    best_F = F[:]
                improved = True
            else:
                F_set.add(old_m)
    
    return best_F, best_score

def construct():
    N = 12
    M = 40
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    best_F = None
    best_score = -1
    
    # Multiple restarts with chain-based init + local search + B&B polish
    num_restarts = 0
    while time.time() < deadline:
        # Chain-based initialization
        F = _chain_based_initial()
        num_restarts += 1
        
        # Local search
        remaining = deadline - time.time()
        if remaining < 3:
            break
        
        if num_restarts <= 3:
            ls_time = min(remaining - 5, 8)
        elif num_restarts <= 6:
            ls_time = min(remaining - 5, 4)
        else:
            ls_time = min(remaining - 5, 2)
        
        if ls_time < 0.5:
            break
        
        F, score = _local_search(F, ls_time)
        
        if score > best_score:
            best_score = score
            best_F = F[:]
        
        # B&B polish on best found so far
        remaining = deadline - time.time()
        if remaining > 3 and best_F is not None:
            bb_time = min(remaining - 2, 3)
            if bb_time > 0.5:
                F, score = _branch_and_bound_polish(best_F, bb_time)
                if score > best_score:
                    best_score = score
                    best_F = F[:]
        
        # Check if we've hit a good plateau
        if best_score >= 300 and num_restarts >= 3:
            # Spend remaining time on more polishing
            remaining = deadline - time.time()
            if remaining > 2:
                F, score = _local_search(best_F, min(remaining - 1, 10))
                if score > best_score:
                    best_score = score
                    best_F = F[:]
            remaining = deadline - time.time()
            if remaining > 1:
                F, score = _branch_and_bound_polish(best_F, max(0, remaining - 0.5))
                if score > best_score:
                    best_score = score
                    best_F = F[:]
            break
    
    # If we never ran local search, at least use the chain-based result
    if best_F is None:
        best_F = _chain_based_initial()
    
    # Final polish: one more local search pass on the best found
    if time.time() < deadline - 1:
        remaining = deadline - 1 - time.time()
        if remaining > 0.5:
            best_F, best_score = _local_search(best_F, remaining)
        
        # Final B&B polish
        remaining = deadline - 0.5 - time.time()
        if remaining > 0.5:
            best_F, best_score = _branch_and_bound_polish(best_F, remaining)
    
    # Convert bitmasks to list of lists of integers
    result = []
    for mask in best_F:
        subset = [i for i in range(N) if (mask >> i) & 1]
        result.append(subset)
    
    return result
