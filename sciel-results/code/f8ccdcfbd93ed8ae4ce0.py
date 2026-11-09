import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    n = 12
    m = 40
    num_masks = 1 << n  # 4096
    
    # Precompute comparability: for each mask s, which other masks t are comparable (one is strict subset of other)
    # comp_bits[s] is a bitmask where bit t is set if s and t are comparable
    # This is 4096 * 4096 bits = 2MB, very manageable
    # But we can't store as Python int easily for 4096 bits... actually we can, Python ints are arbitrary precision
    # Let's use a list of Python ints, each 4096 bits
    
    # Actually, for faster access, let's precompute for each mask s, a list of comparable masks
    # But that's 4096 * ~variable_length lists. Let's just use direct check in the inner loop.
    
    # Better: precompute comp_bits as Python ints
    comp_bits = [0] * num_masks
    for s in range(num_masks):
        cb = 0
        for t in range(num_masks):
            if s == t:
                continue
            if (s & t) == s or (s & t) == t:
                cb |= (1 << t)
        comp_bits[s] = cb
    
    def count_comparables(family):
        count = 0
        k = len(family)
        for i in range(k):
            fi = family[i]
            for j in range(i + 1, k):
                fj = family[j]
                if (fi & fj) == fi or (fi & fj) == fj:
                    count += 1
        return count
    
    def greedy_init():
        """Build initial family using greedy approach with multiple diverse orderings"""
        family = []
        family_set = set()
        
        # Start with empty set and full set
        family.append(0)
        family_set.add(0)
        family.append((1 << n) - 1)
        family_set.add((1 << n) - 1)
        
        # Greedily add sets that maximize comparability with existing family
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
    
    # Simulated Annealing with optimized delta computation
    T = 10.0
    T_min = 0.01
    alpha = 0.99995  # Very slow cooling
    max_iter_per_restart = 300000
    num_restarts = 5
    
    best_family = family[:]
    best_score = current_score
    
    family_set = set(family)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        T = 10.0
        current_score = count_comparables(family)
        
        for it in range(max_iter_per_restart):
            if time.time() > deadline - 5.0:
                break
            
            i = random.randint(0, m - 1)
            old_s = family[i]
            
            # Smart candidate generation: bias toward sets comparable to existing family members
            if random.random() < 0.75:
                ref_idx = random.randint(0, m - 1)
                if ref_idx == i:
                    ref_idx = (ref_idx + 1) % m
                ref_s = family[ref_idx]
                
                if random.random() < 0.5:
                    # Superset of ref_s
                    new_s = ref_s
                    num_add = random.randint(1, 3)
                    for _ in range(num_add):
                        bit = random.randint(0, n - 1)
                        new_s |= (1 << bit)
                else:
                    # Subset of ref_s
                    new_s = ref_s
                    num_remove = random.randint(1, 3)
                    for _ in range(num_remove):
                        bit = random.randint(0, n - 1)
                        new_s &= ~(1 << bit)
            else:
                new_s = random.randint(0, num_masks - 1)
            
            if new_s == old_s:
                continue
            
            if new_s in family_set:
                continue
            
            # Compute delta: change in number of comparable pairs
            count_new = 0
            count_old = 0
            
            for j, f in enumerate(family):
                if j == i:
                    continue
                if (f & new_s) == f or (f & new_s) == new_s:
                    count_new += 1
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
    
    # Final exhaustive local search polish
    if time.time() < deadline - 3.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(m):
                old_s = family[i]
                best_local_gain = 0
                best_local_s = -1
                
                # Try all 4096 possible masks as candidates
                for new_s in range(num_masks):
                    if new_s == old_s:
                        continue
                    if new_s in family_set:
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
    
    result = []
    for s in family:
        subset = []
        for i in range(n):
            if (s >> i) & 1:
                subset.append(i)
        result.append(subset)
        
    return result
