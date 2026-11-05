import time
import random
import math
from functools import lru_cache

# Precompute Smallest Prime Factors (SPF) for fast factorization
MAX_VAL = 20000
spf = [0] * (MAX_VAL + 1)
for i in range(2, MAX_VAL + 1):
    if spf[i] == 0:
        for j in range(i, MAX_VAL + 1, i):
            if spf[j] == 0:
                spf[j] = i

def get_prime_factors(n):
    factors = set()
    if n < 2:
        return factors
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def is_smooth(n, max_prime):
    """Check if n has no prime factors greater than max_prime"""
    if n < 2:
        return True
    while n > 1:
        p = spf[n]
        if p > max_prime:
            return False
        while n % p == 0:
            n //= p
    return True

# Candidate prime sets to try, in order of increasing size
# S = {2,3,5} means all sums should be 5-smooth
# S = {2,3,5,7} means all sums should be 7-smooth
# etc.

def try_residue_class_approach(prime_set, time_limit=5.0):
    """
    Try to find 12 numbers a_i such that all pairwise sums a_i + a_j are S-smooth,
    where S is the given prime set.
    
    Uses the fact that if a_i + a_j is S-smooth, then a_i + a_j mod p = 0 for all p in S
    is NOT required - we just need the sum to be S-smooth.
    
    Instead, we use backtracking: pick values one at a time, ensuring that when we add
    a new value, its sum with all previous values is S-smooth.
    """
    start = time.time()
    deadline = start + time_limit
    
    max_prime = max(prime_set)
    
    # Precompute: for a given residue r mod M, which values v in [1, 10000] 
    # have v % M == r AND v is such that v + x is S-smooth for various x?
    # This is complex. Instead, let's do direct backtracking with pruning.
    
    # We'll search for values in a range. Use a_i = base + step * k_i for some base, step
    # to ensure distinctness and keep values small.
    
    # Try different parameterizations
    best_solution = None
    best_cost = float('inf')
    
    # Parameterization: a_i = offset + stride * k_i
    # k_i are distinct non-negative integers
    # a_i + a_j = 2*offset + stride*(k_i + k_j)
    
    # For the sum to be S-smooth, we need 2*offset + stride*(k_i + k_j) to be S-smooth
    # for all pairs i < j.
    
    # Let's try small strides and offsets
    for stride in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30]:
        if time.time() > deadline:
            break
        
        for offset in range(1, 50):
            if time.time() > deadline:
                break
            
            # Check if 2*offset is S-smooth (this is the base)
            base = 2 * offset
            if not is_smooth(base, max_prime):
                continue
            
            # Now we need stride*(k_i + k_j) to be such that base + stride*(k_i + k_j) is S-smooth
            # for all pairs.
            
            # Precompute valid k values: k is valid if for all other selected k', 
            # base + stride*(k + k') is S-smooth.
            
            # Let's do backtracking
            selected_k = []
            used_k = set()
            
            def backtrack(depth):
                nonlocal best_solution, best_cost
                if time.time() > deadline:
                    return False
                
                if depth == 12:
                    # Found a solution!
                    # Compute actual cost (should be 0 new primes beyond S)
                    vals = [offset + stride * k for k in selected_k]
                    # Verify all sums are S-smooth
                    all_smooth = True
                    for i in range(12):
                        for j in range(i+1, 12):
                            s = vals[i] + vals[j]
                            if not is_smooth(s, max_prime):
                                all_smooth = False
                                break
                        if not all_smooth:
                            break
                    if all_smooth:
                        best_solution = vals[:]
                        best_cost = 0
                        return True
                    return False
                
                # Try candidate k values
                # Start from small k values
                for k in range(0, 200):
                    if time.time() > deadline:
                        return False
                    
                    if k in used_k:
                        continue
                    
                    # Check if k can be added: for all existing k', base + stride*(k + k') is S-smooth
                    can_add = True
                    for k_existing in selected_k:
                        s_val = base + stride * (k + k_existing)
                        if s_val > 10000 or s_val < 2:
                            can_add = False
                            break
                        if not is_smooth(s_val, max_prime):
                            can_add = False
                            break
                    
                    if not can_add:
                        continue
                    
                    # Also check that the value itself is in range
                    val = offset + stride * k
                    if val < 1 or val > 10000:
                        continue
                    
                    selected_k.append(k)
                    used_k.add(k)
                    
                    if backtrack(depth + 1):
                        return True
                    
                    selected_k.pop()
                    used_k.discard(k)
                
                return False
            
            if backtrack(0):
                if best_cost == 0:
                    return best_solution
    
    return None

def count_total_primes(a_vals):
    """Count total distinct prime factors of all pairwise sums"""
    primes = set()
    n = len(a_vals)
    for i in range(n):
        for j in range(i + 1, n):
            s = a_vals[i] + a_vals[j]
            pf = get_prime_factors(s)
            primes.update(pf)
    return len(primes)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin within 90s limit

    # First, try the residue class / smooth sum approach with small prime sets
    # S = {2, 3, 5}: all sums should be 5-smooth
    # S = {2, 3, 5, 7}: all sums should be 7-smooth
    # etc.
    
    # Try increasingly larger prime sets
    prime_sets = [
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
    ]
    
    best_result = None
    best_score = float('inf')
    
    for prime_set in prime_sets:
        if time.time() > deadline - 10.0:
            break
        
        sol = try_residue_class_approach(prime_set, time_limit=8.0)
        if sol is not None:
            score = count_total_primes(sol)
            if score < best_score:
                best_score = score
                best_result = sol
            
            # If we found a solution with few primes, we can stop early
            if score <= len(prime_set):
                break
    
    # If the smooth approach didn't find a good solution, fall back to simulated annealing
    if best_result is None or best_score > 5:
        # Use simulated annealing on the k-parameterization
        # a_i = offset + stride * k_i
        
        # Try multiple parameterizations
        best_k = None
        best_k_cost = float('inf')
        
        for stride in [10, 15, 20, 30]:
            if time.time() > deadline - 5.0:
                break
            
            for offset in [1, 5, 10, 15, 20, 25, 30]:
                if time.time() > deadline - 5.0:
                    break
                
                # Initial solution
                k_vals = list(range(12))
                a_vals = [offset + stride * k for k in k_vals]
                
                # Check bounds
                if any(a < 1 or a > 10000 for a in a_vals):
                    continue
                
                current_cost = count_total_primes(a_vals)
                best_local_k = k_vals[:]
                best_local_cost = current_cost
                
                T = 3.0
                cooling_rate = 0.999
                
                while time.time() < deadline - 5.0:
                    if T < 0.01:
                        T = 1.0
                    
                    idx = random.randint(0, 11)
                    old_val = k_vals[idx]
                    
                    if random.random() < 0.1:
                        new_val = random.randint(0, 300)
                    else:
                        step = random.choice([1, 2, 3, 5])
                        new_val = max(0, min(330, old_val + random.randint(-step, step)))
                    
                    if new_val in k_vals and new_val != old_val:
                        continue
                    
                    k_vals[idx] = new_val
                    a_vals = [offset + stride * k for k in k_vals]
                    
                    # Check bounds
                    if any(a < 1 or a > 10000 for a in a_vals):
                        k_vals[idx] = old_val
                        continue
                    
                    new_cost = count_total_primes(a_vals)
                    
                    if new_cost < current_cost:
                        current_cost = new_cost
                        if new_cost < best_local_cost:
                            best_local_cost = new_cost
                            best_local_k = k_vals[:]
                    else:
                        delta = new_cost - current_cost
                        if delta == 0:
                            if random.random() < 0.8:
                                current_cost = new_cost
                            else:
                                k_vals[idx] = old_val
                        else:
                            if random.random() < math.exp(-delta / T):
                                current_cost = new_cost
                            else:
                                k_vals[idx] = old_val
                    
                    T *= cooling_rate
                
                if best_local_cost < best_k_cost:
                    best_k_cost = best_local_cost
                    best_k = best_local_k
                    best_stride = stride
                    best_offset = offset
        
        if best_k is not None:
            result = [best_offset + best_stride * k for k in best_k]
            score = count_total_primes(result)
            if score < best_score:
                best_score = score
                best_result = result
    
    # Final fallback: if we still don't have a good result, use a simple heuristic
    if best_result is None:
        # Use small consecutive numbers
        best_result = list(range(1, 13))
    
    # Ensure distinctness and bounds
    best_result = list(set(best_result))
    if len(best_result) < 12:
        # Pad with larger numbers
        existing = set(best_result)
        for i in range(13, 10000):
            if len(best_result) >= 12:
                break
            if i not in existing:
                best_result.append(i)
                existing.add(i)
    
    best_result = best_result[:12]
    best_result.sort()
    
    return best_result
# EVOLVE-BLOCK-END
