"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 70  # safety margin before 90s kill

    MAX_N = 256

    # Generate all primes up to MAX_N using Sieve of Eratosthenes
    def get_primes(n):
        if n < 2:
            return []
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i in range(2, n + 1) if sieve[i]]

    primes = get_primes(MAX_N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    n_primes = len(primes)

    # For each number 2..256, compute its prime factor set (as a bitmask)
    # and determine which numbers are "useful" (not dominated by another number
    # with the same or subset of prime factors and higher value)
    
    # A number n is dominated if there exists m > n with prime_factors(m) ⊆ prime_factors(n)
    # In that case, m is always at least as good as n for any feasible solution.
    
    # Compute prime factor masks for all numbers 2..256
    num_masks = {}  # n -> bitmask of prime factors
    for n in range(2, MAX_N + 1):
        mask = 0
        temp = n
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << i)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            # temp is a prime
            if temp in prime_to_idx:
                mask |= (1 << prime_to_idx[temp])
            else:
                # Should not happen since we have all primes up to 256
                pass
        num_masks[n] = mask

    # Find non-dominated numbers: n is dominated if exists m > n with mask(m) ⊆ mask(n)
    # For efficiency, group by mask
    from collections import defaultdict
    mask_to_nums = defaultdict(list)
    for n in range(2, MAX_N + 1):
        mask_to_nums[num_masks[n]].append(n)

    # For each mask, only the largest number matters (others are dominated)
    # But also, a number with mask M is dominated by any number with mask M' ⊆ M and value > n
    # This is complex, so let's just keep the max for each mask and then do a subset check
    
    # Candidate numbers: for each distinct mask, take the largest number with that mask
    candidates = []
    for mask, nums in mask_to_nums.items():
        if nums:
            candidates.append(max(nums))

    # Further prune: remove candidates that are dominated by another candidate
    # n is dominated by m if mask(m) ⊆ mask(n) and m > n
    candidates_set = set(candidates)
    final_candidates = []
    for n in candidates:
        dominated = False
        mask_n = num_masks[n]
        for m in candidates:
            if m > n:
                mask_m = num_masks[m]
                if (mask_m & mask_n) == mask_m:  # mask_m ⊆ mask_n
                    dominated = True
                    break
        if not dominated:
            final_candidates.append(n)

    # Now solve: select a subset of final_candidates with disjoint prime masks, maximizing sum
    # Plus we always include 1
    # This is a weighted set packing problem
    
    # Use branch and bound / backtracking with pruning
    # Sort candidates by value descending for better pruning
    final_candidates.sort(key=lambda x: -x)
    n_cands = len(final_candidates)
    
    # Precompute masks
    cand_masks = [num_masks[n] for n in final_candidates]
    
    best_sum = 0
    best_selection = []
    
    # Branch and bound
    def bb(idx, used_mask, current_sum, current_sel):
        nonlocal best_sum, best_selection
        if time.time() > deadline - 2:
            return
        if idx == n_cands:
            if current_sum > best_sum:
                best_sum = current_sum
                best_selection = list(current_sel)
            return
        
        # Upper bound: current_sum + sum of remaining candidates that don't conflict
        ub = current_sum
        for i in range(idx, n_cands):
            if (cand_masks[i] & used_mask) == 0:
                ub += final_candidates[i]
            if ub <= best_sum:
                break
        
        if ub <= best_sum:
            return
        
        # Option 1: skip candidate idx
        bb(idx + 1, used_mask, current_sum, current_sel)
        
        # Option 2: take candidate idx if no conflict
        if (cand_masks[idx] & used_mask) == 0:
            current_sel.append(final_candidates[idx])
            bb(idx + 1, used_mask | cand_masks[idx], current_sum + final_candidates[idx], current_sel)
            current_sel.pop()
    
    bb(0, 0, 0, [])
    
    result = [1] + best_selection
    result = [int(x) for x in result]
    
    # Verify pairwise coprimality
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: greedy prime powers
        result = [1]
        for p in primes:
            power = p
            while power * p <= MAX_N:
                power *= p
            result.append(power)
        result = [int(x) for x in result]
    
    return result
# EVOLVE-BLOCK-END
