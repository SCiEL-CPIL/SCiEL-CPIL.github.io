import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 256
    
    # Precompute factor masks
    primes = []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    factor_mask = [0] * (n + 1)
    for i in range(2, n + 1):
        mask = 0
        x = i
        for p in primes:
            if p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_to_bit[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_to_bit[x])
        factor_mask[i] = mask
    
    # Precompute pairwise compatibility: compat[a][b] = True if gcd(a,b)==1
    # But that's 256x256 = 65536 entries, which is fine
    # Actually, let's precompute for each value, the set of values it's compatible with
    # Or better: for each value, precompute a bitmask of which other values conflict
    
    # Let's use a different approach: precompute conflict_mask[val] as a bitset of values 1..256 that conflict with val
    conflict_mask = [0] * (n + 1)
    for a in range(1, n + 1):
        fa = factor_mask[a]
        if fa == 0:
            continue
        for b in range(1, n + 1):
            if a == b:
                continue
            fb = factor_mask[b]
            if fb == 0:
                continue
            if (fa & fb) != 0:
                conflict_mask[a] |= (1 << b)
    
    def is_compatible(val, current_set):
        """Check if val is compatible with all elements in current_set"""
        cm = conflict_mask[val]
        if cm == 0:
            return True
        # Check if any element in current_set conflicts with val
        # current_set is a set of ints
        for v in current_set:
            if (cm >> v) & 1:
                return False
        return True
    
    def local_search(initial_set, time_limit):
        best_set = set(initial_set)
        best_sum = sum(best_set)
        candidates = list(range(n, 1, -1))  # Don't include 1 in candidates for swaps
        end_time = time.time() + time_limit
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            
            # Phase 1: Add any compatible value
            for val in range(n, 1, -1):
                if time.time() >= end_time:
                    break
                if val in best_set:
                    continue
                if is_compatible(val, best_set):
                    best_set.add(val)
                    best_sum += val
                    improved = True
            
            # Phase 2: Swap 1-for-1 (remove one, add one larger)
            for remove_val in sorted(best_set, key=lambda x: -x):  # Try removing larger values first
                if time.time() >= end_time:
                    break
                # Only consider removing if there's potential for improvement
                # Find best add_val that's compatible with best_set - {remove_val}
                reduced_set = best_set - {remove_val}
                best_add = None
                best_add_val = 0
                for add_val in range(n, 1, -1):
                    if time.time() >= end_time:
                        break
                    if add_val in reduced_set:
                        continue
                    if add_val <= remove_val:
                        continue  # Only consider strictly better swaps
                    if is_compatible(add_val, reduced_set):
                        best_add = add_val
                        break  # First compatible is best since we iterate descending
                
                if best_add is not None:
                    best_set = reduced_set
                    best_set.add(best_add)
                    best_sum = best_sum - remove_val + best_add
                    improved = True
            
            # Phase 3: Swap 1-for-2 (remove one, add two)
            if time.time() < end_time:
                for remove_val in sorted(best_set, key=lambda x: -x):
                    if time.time() >= end_time:
                        break
                    
                    reduced_set = best_set - {remove_val}
                    reduced_sum = best_sum - remove_val
                    
                    # Try to find two values that together are better than remove_val
                    # and are compatible with reduced_set and each other
                    best_pair = None
                    best_pair_sum = 0
                    
                    for add1 in range(n, 1, -1):
                        if time.time() >= end_time:
                            break
                        if add1 in reduced_set:
                            continue
                        if not is_compatible(add1, reduced_set):
                            continue
                        
                        # Now try to find add2 compatible with reduced_set + {add1}
                        temp_set = reduced_set | {add1}
                        for add2 in range(n, 1, -1):
                            if time.time() >= end_time:
                                break
                            if add2 in temp_set:
                                continue
                            if add2 <= add1:
                                continue  # Avoid duplicates
                            if not is_compatible(add2, temp_set):
                                continue
                            
                            pair_sum = add1 + add2
                            if pair_sum > remove_val and pair_sum > best_pair_sum:
                                best_pair = (add1, add2)
                                best_pair_sum = pair_sum
                            break  # First compatible add2 for this add1 is best (descending order)
                    
                    if best_pair is not None and best_pair_sum > remove_val:
                        best_set = reduced_set
                        best_set.add(best_pair[0])
                        best_set.add(best_pair[1])
                        best_sum = reduced_sum + best_pair_sum
                        improved = True
            
            # Phase 4: Swap 2-for-3 (remove two, add three) - only if time permits
            if time.time() < end_time - 2.0:
                for remove1 in sorted(best_set, key=lambda x: -x):
                    if time.time() >= end_time:
                        break
                    for remove2 in sorted(best_set - {remove1}, key=lambda x: -x):
                        if time.time() >= end_time:
                            break
                        
                        reduced_set = best_set - {remove1, remove2}
                        reduced_sum = best_sum - remove1 - remove2
                        threshold = remove1 + remove2
                        
                        # Try to find three values that sum to more than threshold
                        best_triple = None
                        best_triple_sum = 0
                        
                        for add1 in range(n, 1, -1):
                            if time.time() >= end_time:
                                break
                            if add1 in reduced_set:
                                continue
                            if not is_compatible(add1, reduced_set):
                                continue
                            
                            temp_set1 = reduced_set | {add1}
                            for add2 in range(n, 1, -1):
                                if time.time() >= end_time:
                                    break
                                if add2 in temp_set1:
                                    continue
                                if add2 <= add1:
                                    continue
                                if not is_compatible(add2, temp_set1):
                                    continue
                                
                                temp_set2 = temp_set1 | {add2}
                                for add3 in range(n, 1, -1):
                                    if time.time() >= end_time:
                                        break
                                    if add3 in temp_set2:
                                        continue
                                    if add3 <= add2:
                                        continue
                                    if not is_compatible(add3, temp_set2):
                                        continue
                                    
                                    triple_sum = add1 + add2 + add3
                                    if triple_sum > threshold and triple_sum > best_triple_sum:
                                        best_triple = (add1, add2, add3)
                                        best_triple_sum = triple_sum
                                    break
                        
                        if best_triple is not None and best_triple_sum > threshold:
                            best_set = reduced_set
                            best_set.add(best_triple[0])
                            best_set.add(best_triple[1])
                            best_set.add(best_triple[2])
                            best_sum = reduced_sum + best_triple_sum
                            improved = True
                            break

        return best_set, best_sum

    # Generate multiple initial solutions
    
    # 1. Prime Power Heuristic
    init_set_1 = {1}
    for p in primes:
        power = p
        while power * p <= n:
            power *= p
        if is_compatible(power, init_set_1):
            init_set_1.add(power)
            
    # 2. Greedy Largest Heuristic
    init_set_2 = {1}
    for val in range(n, 1, -1):
        if is_compatible(val, init_set_2):
            init_set_2.add(val)
            
    # 3. Greedy with prime powers prioritized
    init_set_3 = {1}
    # First add prime powers
    for p in primes:
        power = p
        while power * p <= n:
            power *= p
        if is_compatible(power, init_set_3):
            init_set_3.add(power)
    # Then add remaining values greedily
    for val in range(n, 1, -1):
        if val not in init_set_3 and is_compatible(val, init_set_3):
            init_set_3.add(val)
            
    best_overall_set = set()
    best_overall_sum = 0
    
    # Allocate time: 30% for Init 1, 30% for Init 2, 25% for Init 3, 15% for Random Restarts
    total_time = 80.0
    t1 = total_time * 0.30
    t2 = total_time * 0.30
    t3 = total_time * 0.25
    t4 = total_time * 0.15
    
    # Run local search on Init 1
    s1, sum1 = local_search(init_set_1, t1)
    if sum1 > best_overall_sum:
        best_overall_set = s1
        best_overall_sum = sum1
        
    # Run local search on Init 2
    s2, sum2 = local_search(init_set_2, t2)
    if sum2 > best_overall_sum:
        best_overall_set = s2
        best_overall_sum = sum2
        
    # Run local search on Init 3
    if time.time() < deadline - 5.0:
        s3, sum3 = local_search(init_set_3, t3)
        if sum3 > best_overall_sum:
            best_overall_set = s3
            best_overall_sum = sum3
    
    # Random Restarts
    if time.time() < deadline - 5.0:
        # Do as many random restarts as possible with at least 3s each
        min_restart_time = 3.0
        max_restarts = int((deadline - 5.0 - time.time()) / min_restart_time)
        max_restarts = max(1, min(max_restarts, 5))
        
        for _ in range(max_restarts):
            if time.time() >= deadline - 5.0:
                break
            
            rand_set = {1}
            cands = list(range(n, 1, -1))
            random.shuffle(cands)
            
            for val in cands:
                if is_compatible(val, rand_set):
                    rand_set.add(val)
                    
            rs, rsum = local_search(rand_set, 3.0)
            if rsum > best_overall_sum:
                best_overall_set = rs
                best_overall_sum = rsum

    return [int(x) for x in best_overall_set]
