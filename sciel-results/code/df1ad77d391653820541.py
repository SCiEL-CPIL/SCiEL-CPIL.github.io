import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s limit
    
    n = 12
    m = 40
    num_masks = 1 << n  # 4096
    
    # Precompute comparability: for each mask s, count how many masks in a given set are comparable
    # We'll use a different approach: precompute for each mask, its "comparable count" contribution
    
    # For efficient delta calculation, precompute for each mask s:
    # - list of all masks that are strict subsets of s
    # - list of all masks that are strict supersets of s
    # Then for a family, count = |{f in family: f is subset of s or s is subset of f}|
    
    # Actually, let's precompute a boolean matrix but use it more efficiently
    # comp[s][t] = 1 if s is a strict subset of t, 0 otherwise
    # Then s and t are comparable if comp[s][t] or comp[t][s]
    
    # Precompute subset relation
    # subset_of[s] = list of masks that are strict subsets of s
    # superset_of[s] = list of masks that are strict supersets of s
    
    subset_of = [[] for _ in range(num_masks)]
    superset_of = [[] for _ in range(num_masks)]
    
    for s in range(num_masks):
        # Find all strict subsets of s
        sub = s
        while sub:
            subset_of[s].append(sub)
            sub = (sub - 1) & s
        # Find all strict supersets of s
        # Supersets of s are masks t where (t & s) == s and t != s
        # Equivalently, t = s | u where u is a subset of (~s) & (num_masks-1) and u != 0
        free_bits = (num_masks - 1) ^ s
        u = free_bits
        while u:
            superset_of[s].append(s | u)
            u = (u - 1) & free_bits
    
    def count_comparables(family):
        """Count unordered pairs where one is a strict subset of the other."""
        count = 0
        k = len(family)
        for i in range(k):
            fi = family[i]
            for j in range(i + 1, k):
                fj = family[j]
                if (fi & fj) == fi or (fi & fj) == fj:
                    count += 1
        return count
    
    def compute_comparable_counts(family):
        """For each family member, count how many other family members it's comparable with."""
        counts = [0] * len(family)
        k = len(family)
        for i in range(k):
            fi = family[i]
            for j in range(k):
                if i == j:
                    continue
                fj = family[j]
                if (fi & fj) == fi or (fi & fj) == fj:
                    counts[i] += 1
        return counts
    
    def greedy_init():
        """Build initial family using greedy approach with multiple chains."""
        family = []
        family_set = set()
        
        # Start with empty set and full set
        family.append(0)
        family_set.add(0)
        family.append(num_masks - 1)
        family_set.add(num_masks - 1)
        
        while len(family) < m:
            best_candidate = -1
            best_gain = -1
            
            for s in range(num_masks):
                if s in family_set:
                    continue
                
                gain = 0
                for f in family:
                    if (f & s) == f or (f & s) == s:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = s
            
            if best_candidate != -1:
                family.append(best_candidate)
                family_set.add(best_candidate)
        
        return family
    
    family = greedy_init()
    current_score = count_comparables(family)
    
    best_family = family[:]
    best_score = current_score
    
    # Maintain comparable counts for fast delta
    comp_counts = compute_comparable_counts(family)
    family_set = set(family)
    
    # Simulated Annealing
    T = 15.0
    T_min = 0.01
    alpha = 0.99995  # Very slow cooling
    max_iter = 500000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        i = random.randint(0, m - 1)
        old_s = family[i]
        
        # Smart candidate generation
        r = random.random()
        if r < 0.5:
            # Generate subset or superset of a random family member
            ref_idx = random.randint(0, m - 1)
            if ref_idx == i:
                ref_idx = (ref_idx + 1) % m
            ref_s = family[ref_idx]
            
            if random.random() < 0.5:
                # Superset
                free_bits = (num_masks - 1) ^ ref_s
                if free_bits == 0:
                    new_s = ref_s
                else:
                    # Pick random non-empty subset of free bits
                    u = free_bits
                    num_remove = random.randint(1, 3)
                    for _ in range(num_remove):
                        if u == 0:
                            break
                        # Remove a random bit from u
                        bits = []
                        temp = u
                        while temp:
                            lb = temp & (-temp)
                            bits.append(lb)
                            temp ^= lb
                        if bits:
                            u ^= random.choice(bits)
                    new_s = ref_s | u if u else ref_s
            else:
                # Subset
                if ref_s == 0:
                    new_s = 0
                else:
                    # Pick random non-empty subset of ref_s
                    sub = ref_s
                    num_remove = random.randint(1, 3)
                    for _ in range(num_remove):
                        if sub == 0:
                            break
                        bits = []
                        temp = sub
                        while temp:
                            lb = temp & (-temp)
                            bits.append(lb)
                            temp ^= lb
                        if bits:
                            sub ^= random.choice(bits)
                    new_s = sub
        elif r < 0.8:
            # Generate by adding/removing 1-3 bits from old_s
            new_s = old_s
            num_change = random.randint(1, 3)
            for _ in range(num_change):
                bit = random.randint(0, n - 1)
                if random.random() < 0.5:
                    new_s |= (1 << bit)
                else:
                    new_s &= ~(1 << bit)
        else:
            new_s = random.randint(0, num_masks - 1)
        
        if new_s == old_s or new_s in family_set:
            continue
        
        # Compute delta efficiently
        # delta = (new comparable count for new_s) - (old comparable count for old_s)
        # But we need to be careful: when we replace old_s with new_s, the counts change
        
        # Count how many family members (excluding i) are comparable to new_s
        count_new = 0
        for j, f in enumerate(family):
            if j == i:
                continue
            if (f & new_s) == f or (f & new_s) == new_s:
                count_new += 1
        
        # Count how many family members (excluding i) are comparable to old_s
        count_old = 0
        for j, f in enumerate(family):
            if j == i:
                continue
            if (f & old_s) == f or (f & old_s) == old_s:
                count_old += 1
        
        delta = count_new - count_old
        
        if delta >= 0 or random.random() < math.exp(delta / T):
            family[i] = new_s
            family_set.remove(old_s)
            family_set.add(new_s)
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_family = family[:]
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    # Final local search polish
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for i in range(m):
                old_s = family[i]
                best_local_gain = 0
                best_local_s = -1
                
                # Try systematic candidates
                candidates = set()
                
                # All immediate supersets
                for bit in range(n):
                    if not (old_s & (1 << bit)):
                        candidates.add(old_s | (1 << bit))
                
                # All immediate subsets
                for bit in range(n):
                    if old_s & (1 << bit):
                        candidates.add(old_s & ~(1 << bit))
                
                # Some random candidates
                for _ in range(15):
                    candidates.add(random.randint(0, num_masks - 1))
                
                for new_s in candidates:
                    if new_s == old_s or new_s in family_set:
                        continue
                    
                    count_new = 0
                    for j, f in enumerate(family):
                        if j == i:
                            continue
                        if (f & new_s) == f or (f & new_s) == new_s:
                            count_new += 1
                    
                    count_old = 0
                    for j, f in enumerate(family):
                        if j == i:
                            continue
                        if (f & old_s) == f or (f & old_s) == old_s:
                            count_old += 1
                    
                    local_gain = count_new - count_old
                    if local_gain > best_local_gain:
                        best_local_gain = local_gain
                        best_local_s = new_s
                
                if best_local_s != -1 and best_local_gain > 0:
                    family[i] = best_local_s
                    family_set.remove(old_s)
                    family_set.add(best_local_s)
                    current_score += best_local_gain
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_family = family[:]
    
    family = best_family
    
    # Convert to list of lists of integers
    result = []
    for s in family:
        subset = []
        for i in range(n):
            if (s >> i) & 1:
                subset.append(i)
        result.append(subset)
    
    return result
