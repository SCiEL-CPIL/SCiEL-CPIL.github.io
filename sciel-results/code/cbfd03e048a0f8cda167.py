import time
import numpy as np
import random
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def _is_dissociated_int(S, recips_num, recips_den):
    """
    Check if set S is dissociated using integer arithmetic.
    For each element s in S, we have 1/s = recips_num[s] / recips_den[s].
    We compute all subset sums as fractions with a common denominator.
    Two subset sums are equal iff their numerators (scaled to common denom) are equal.
    """
    if not S:
        return True
    
    n = len(S)
    if n <= 1:
        return True
    
    # For small n, use floating point as a quick filter, then verify with integers
    # Actually, let's just use integer arithmetic directly for correctness
    
    # Compute the LCM of all denominators in S
    # This could be large, but for n <= 15 or so, it's manageable
    # Actually, the LCM of [1..22] is about 232792560, which fits in a 64-bit int
    
    # Instead of computing LCM of denominators, we can use a different approach:
    # Represent each reciprocal as a fraction, and track subset sums as (num, den) pairs
    # But that's slow. Better: use the fact that 1/s for s in [1,22] are all rational.
    # The LCM of 1..22 is 232792560.
    # So 1/s = (232792560 // s) / 232792560
    # All subset sums are multiples of 1/232792560, so we can just track the numerator!
    
    # Precompute: for each s, the numerator when expressed with denominator LCM(1..22)
    # LCM(1..22) = 232792560
    # But we only need the LCM of the denominators in S, which is smaller.
    # However, using the global LCM is simpler and the numbers still fit.
    
    # Actually, let's use a smarter approach:
    # For a set S, compute D = LCM of all elements in S.
    # Then 1/s = D/s for s in S (as integer numerators with common denominator D).
    # All subset sums are sums of these integers, so we just need to check
    # if all subset sums of {D/s for s in S} are distinct.
    
    D = _lcm_list(S)
    # Now check if all subset sums of [D//s for s in S] are distinct
    vals = [D // s for s in S]
    
    # Use a set to track subset sums
    sums = {0}
    for v in vals:
        new_sums = set()
        for s in sums:
            ns = s + v
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
    
    return True

def _is_dissociated_fast(S, precomputed):
    """
    Fast dissociation check using precomputed LCM numerators.
    precomputed[s] = LCM(1..22) // s, so all reciprocals share a common denominator.
    """
    if not S:
        return True
    if len(S) <= 1:
        return True
    
    vals = [precomputed[s] for s in S]
    
    sums = {0}
    for v in vals:
        new_sums = set()
        for s in sums:
            ns = s + v
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin for 90s limit
    
    # Precompute: LCM(1..22) = 232792560
    # For each s in [1,22], precomputed[s] = 232792560 // s
    LCM_22 = 232792560
    precomputed = {i: LCM_22 // i for i in range(1, 23)}
    universe = list(range(1, 23))
    
    # Known good seed (size 9)
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    random.seed(42)
    np.random.seed(42)
    
    # Strategy 1: Greedy with various orders
    orders = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
    ]
    
    max_greedy_randoms = 200
    for _ in range(max_greedy_randoms):
        if time.time() > deadline - 30:
            break
        cand_order = list(range(1, 23))
        random.shuffle(cand_order)
        orders.append(cand_order)
    
    for order in orders:
        if time.time() > deadline - 30:
            break
            
        current_set = []
        used = set()
        
        for x in order:
            if x in used:
                continue
            new_set = current_set + [x]
            if _is_dissociated_fast(new_set, precomputed):
                current_set = new_set
                used.add(x)
                
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
        if best_size >= 15:
            break

    # Strategy 2: Local search with swaps
    # Start from best_set and try to grow it
    if time.time() < deadline - 25:
        S = best_set[:]
        used = set(S)
        
        max_local_iters = 10000
        for it in range(max_local_iters):
            if time.time() > deadline - 25:
                break
            
            improved = False
            
            # Phase 1: Try adding any unused element
            for x in universe:
                if x in used:
                    continue
                new_set = S + [x]
                if _is_dissociated_fast(new_set, precomputed):
                    S = new_set
                    used.add(x)
                    improved = True
                    if len(S) > best_size:
                        best_set = S[:]
                        best_size = len(S)
                    break
            
            if improved:
                continue
            
            # Phase 2: Try swap moves
            for i, x in enumerate(S):
                if time.time() > deadline - 25:
                    break
                S_without = S[:i] + S[i+1:]
                used.discard(x)
                for y in universe:
                    if y in used:
                        continue
                    new_set = S_without + [y]
                    if _is_dissociated_fast(new_set, precomputed):
                        S = new_set
                        used.add(y)
                        improved = True
                        if len(S) > best_size:
                            best_set = S[:]
                            best_size = len(S)
                        break
                if improved:
                    break
                S = S_without + [x]
                used.add(x)
            
            if not improved:
                break
    
    # Strategy 3: Random sampling for larger sets
    if time.time() < deadline - 5:
        max_target = min(22, max(best_size + 1, 14))
        
        for target_size in range(max_target, best_size, -1):
            if time.time() > deadline - 5:
                break
            
            max_attempts = 5000
            if target_size >= 14:
                max_attempts = 3000
            elif target_size >= 12:
                max_attempts = 10000
            else:
                max_attempts = 30000
            
            attempts = 0
            while attempts < max_attempts:
                if time.time() > deadline - 3:
                    break
                
                candidate = random.sample(universe, target_size)
                if _is_dissociated_fast(candidate, precomputed):
                    best_set = candidate
                    best_size = target_size
                    break
                attempts += 1
            
            if len(best_set) == target_size:
                break
    
    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
# EVOLVE-BLOCK-END
