import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 12
    m = 40
    total_masks = 1 << n  # 4096 possible subsets
    
    # Precompute popcount for all masks
    popcount = [0] * total_masks
    for mask in range(1, total_masks):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)
    
    # Precompute comparability matrix: comp[a][b] = True if a is strict subset of b
    # This is a 4096x4096 boolean array, but we'll use bitsets for efficiency
    # comp_subset[a] = bitmask of all masks that are strict supersets of a
    # comp_superset[a] = bitmask of all masks that are strict subsets of a
    
    # For each mask, compute all masks that are subsets of it (including itself)
    # Then strict subsets = subsets minus itself
    # And strict supersets: for each mask, all masks that contain it and are different
    
    # More efficient: precompute for each mask, the list of masks comparable with it
    # But 4096*4096 = 16M entries, which is manageable
    
    # Actually, let's precompute a 2D array or use a different approach
    # For a family of size m=40, we only need comparability among family members
    
    # Count comparable pairs in a family (list of masks)
    def count_comparable(family):
        count = 0
        k = len(family)
        for i in range(k):
            a = family[i]
            for j in range(i + 1, k):
                b = family[j]
                if (a & b) == a and a != b:
                    count += 1
                elif (b & a) == b and b != a:
                    count += 1
        return count
    
    # Generate initial solution: use a greedy chain-based approach
    random.seed(42)
    
    # Start with a chain of all sizes 0-12 (13 sets)
    chain = []
    for size in range(13):
        mask = (1 << size) - 1  # {0, 1, ..., size-1}
        chain.append(mask)
    
    # Now add 27 more sets greedily
    family = list(chain)
    
    # Candidate pools: generate random masks
    candidates = list(range(total_masks))
    random.shuffle(candidates)
    
    # Remove masks already in family
    family_set = set(family)
    candidates = [c for c in candidates if c not in family_set]
    
    # Greedily add masks that maximize comparable pairs
    for _ in range(m - len(family)):
        best_mask = None
        best_gain = -1
        # Search more candidates for better initial solution
        for c in candidates[:500]:
            # Compute gain if we add c
            gain = 0
            for f in family:
                if (c & f) == c and c != f:
                    gain += 1
                elif (f & c) == f and f != c:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = c
        if best_mask is not None:
            family.append(best_mask)
            family_set.add(best_mask)
            candidates.remove(best_mask)
    
    # Local search: try to replace one mask with another to improve
    best_score = count_comparable(family)
    best_family = list(family)
    
    # Precompute comparability for current best_family
    # comp[i][j] = True if best_family[i] is strict subset of best_family[j]
    def compute_comparability(fam):
        k = len(fam)
        comp = [[False] * k for _ in range(k)]
        for i in range(k):
            for j in range(k):
                if i != j:
                    a, b = fam[i], fam[j]
                    if (a & b) == a and a != b:
                        comp[i][j] = True
        return comp
    
    # Compute gain/loss efficiently using precomputed comparability
    def compute_net_gain(fam, comp, idx, new_mask, fam_set):
        """Compute net gain of replacing fam[idx] with new_mask"""
        if new_mask in fam_set:
            return -1000000  # Invalid: duplicate
        k = len(fam)
        gain = 0
        loss = 0
        old_mask = fam[idx]
        for j in range(k):
            if j == idx:
                continue
            f = fam[j]
            # Gain: new_mask comparable with f?
            if (new_mask & f) == new_mask and new_mask != f:
                gain += 1
            elif (f & new_mask) == f and f != new_mask:
                gain += 1
            # Loss: old_mask comparable with f?
            if (old_mask & f) == old_mask and old_mask != f:
                loss += 1
            elif (f & old_mask) == f and f != old_mask:
                loss += 1
        return gain - loss
    
    # Try multiple random restarts if time allows
    restarts = 0
    while time.time() < deadline - 8.0:
        restarts += 1
        # Create a perturbed version of best_family
        family = list(best_family)
        
        # Randomly remove 1-2 sets and replace with new ones (less aggressive)
        num_replace = random.randint(1, 2)
        to_replace = random.sample(range(m), num_replace)
        family_set = set(family)
        
        for idx in to_replace:
            # Find a good replacement using systematic search over candidates
            best_mask = None
            best_gain = -1
            # Use a mix of random and structured candidates
            # First try masks close in size to the one being replaced
            old_mask = family[idx]
            old_size = popcount[old_mask]
            
            # Generate candidates: all masks not in family
            # Prioritize masks with similar popcount or that are subsets/supersets
            candidate_list = []
            
            # Add some random candidates
            for _ in range(30):
                c = random.randint(0, total_masks - 1)
                if c not in family_set:
                    candidate_list.append(c)
            
            # Add structured candidates: subsets and supersets of old_mask
            # Subsets of old_mask
            sub = old_mask
            while sub:
                if sub not in family_set and sub != old_mask:
                    candidate_list.append(sub)
                sub = (sub - 1) & old_mask
            # Include empty set if old_mask is not empty
            if old_mask != 0 and 0 not in family_set:
                candidate_list.append(0)
            
            # Supersets of old_mask
            # Generate by adding one bit at a time
            for bit in range(n):
                if not (old_mask >> bit) & 1:
                    new_mask = old_mask | (1 << bit)
                    if new_mask not in family_set and new_mask != old_mask:
                        candidate_list.append(new_mask)
            
            # Try each candidate
            for c in candidate_list:
                if c in family_set:
                    continue
                net_gain = compute_net_gain(family, None, idx, c, family_set)
                if net_gain > best_gain:
                    best_gain = net_gain
                    best_mask = c
            
            if best_mask is not None and best_gain > 0:
                family_set.remove(old_mask)
                family[idx] = best_mask
                family_set.add(best_mask)
        
        score = count_comparable(family)
        if score > best_score:
            best_score = score
            best_family = list(family)
    
    # Final polish: try single replacements with more exhaustive search
    while time.time() < deadline - 2.0:
        improved = False
        family_set = set(best_family)
        for i in range(m):
            # Try to replace best_family[i] with a better mask
            best_mask = None
            best_gain = -1
            old_mask = best_family[i]
            
            # Generate comprehensive candidate list
            candidate_list = []
            
            # Random candidates
            for _ in range(20):
                c = random.randint(0, total_masks - 1)
                if c not in family_set:
                    candidate_list.append(c)
            
            # Subsets of old_mask
            sub = old_mask
            while sub:
                if sub not in family_set and sub != old_mask:
                    candidate_list.append(sub)
                sub = (sub - 1) & old_mask
            if old_mask != 0 and 0 not in family_set:
                candidate_list.append(0)
            
            # Supersets: add one bit
            for bit in range(n):
                if not (old_mask >> bit) & 1:
                    new_mask = old_mask | (1 << bit)
                    if new_mask not in family_set and new_mask != old_mask:
                        candidate_list.append(new_mask)
            
            # Also try masks that share many elements with old_mask
            # XOR-based similarity
            for _ in range(10):
                c = random.randint(0, total_masks - 1)
                if c not in family_set:
                    candidate_list.append(c)
            
            for c in candidate_list:
                if c in family_set:
                    continue
                net_gain = compute_net_gain(best_family, None, i, c, family_set)
                if net_gain > best_gain:
                    best_gain = net_gain
                    best_mask = c
            
            if best_mask is not None and best_gain > 0:
                family_set.remove(old_mask)
                best_family[i] = best_mask
                family_set.add(best_mask)
                improved = True
                break
        if not improved:
            break
    
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
