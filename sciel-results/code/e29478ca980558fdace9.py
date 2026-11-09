"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute prime factors for 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    # prime_factors[1] remains empty
    
    # Find all primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, math.isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # For each prime, find the largest power p^k <= N
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power
        
    # Initial solution: 1 + all prime powers
    initial_set = [1] + list(prime_powers.values())
    
    # Apply specific improvements found by analysis (from inspiration 1)
    # These are known beneficial swaps
    if 121 in initial_set and 23 in initial_set:
        initial_set.remove(121)
        initial_set.remove(23)
        initial_set.append(253)
        
    if 49 in initial_set and 31 in initial_set:
        initial_set.remove(49)
        initial_set.remove(31)
        initial_set.append(217)
        
    if 125 in initial_set and 47 in initial_set:
        initial_set.remove(125)
        initial_set.remove(47)
        initial_set.append(235)
        
    if 81 in initial_set and 83 in initial_set:
        initial_set.remove(81)
        initial_set.remove(83)
        initial_set.append(249)
        
    if 169 in initial_set and 19 in initial_set:
        initial_set.remove(169)
        initial_set.remove(19)
        initial_set.append(247)
    
    # Verify pairwise coprime (sanity check)
    for i in range(len(initial_set)):
        for j in range(i + 1, len(initial_set)):
            if math.gcd(initial_set[i], initial_set[j]) != 1:
                # Fallback: return just prime powers for safety
                return [1] + list(prime_powers.values())
    
    best_sum = sum(initial_set)
    best_set = list(initial_set)
    
    # Local search with swaps (from inspiration 2)
    # We try to improve by replacing subsets of the current set with better subsets
    
    def is_compatible_with_set(num, current_set):
        """Check if num is coprime with all elements in current_set."""
        for x in current_set:
            if math.gcd(num, x) != 1:
                return False
        return True
    
    def is_set_pairwise_coprime(s):
        """Check if all pairs in s are coprime."""
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if math.gcd(s[i], s[j]) != 1:
                    return False
        return True
    
    # Build a list of all numbers 1..N that are NOT in the initial set
    in_set = set(initial_set)
    candidates = [x for x in range(1, N + 1) if x not in in_set]
    
    # Local search: try 1-for-1 swaps first
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_set = list(best_set)
        current_sum = best_sum
        
        for i in range(len(current_set)):
            if time.time() > deadline - 5:
                break
            removed = current_set[i]
            remaining = [x for j, x in enumerate(current_set) if j != i]
            
            # Try adding each candidate
            for cand in candidates:
                if time.time() > deadline - 5:
                    break
                # Check if cand is compatible with remaining
                if is_compatible_with_set(cand, remaining):
                    new_sum = current_sum - removed + cand
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_set = remaining + [cand]
                        improved = True
    
    # Local search: try 2-for-2 swaps
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            current_set = list(best_set)
            current_sum = best_sum
            
            # Try replacing 2 elements with 2 others
            for i in range(len(current_set)):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, len(current_set)):
                    if time.time() > deadline - 10:
                        break
                    removed1 = current_set[i]
                    removed2 = current_set[j]
                    remaining = [x for k, x in enumerate(current_set) if k != i and k != j]
                    
                    # Try adding pairs of candidates
                    for a_idx in range(len(candidates)):
                        if time.time() > deadline - 10:
                            break
                        cand1 = candidates[a_idx]
                        if not is_compatible_with_set(cand1, remaining):
                            continue
                        # Check cand1 with remaining
                        for b_idx in range(a_idx + 1, len(candidates)):
                            cand2 = candidates[b_idx]
                            if not is_compatible_with_set(cand2, remaining):
                                continue
                            if math.gcd(cand1, cand2) != 1:
                                continue
                            new_sum = current_sum - removed1 - removed2 + cand1 + cand2
                            if new_sum > best_sum:
                                best_sum = new_sum
                                best_set = remaining + [cand1, cand2]
                                improved = True
    
    # Convert to list of ints
    result = list(best_set)
    # Ensure all are Python ints
    result = [int(x) for x in result]
    
    return result
# EVOLVE-BLOCK-END
