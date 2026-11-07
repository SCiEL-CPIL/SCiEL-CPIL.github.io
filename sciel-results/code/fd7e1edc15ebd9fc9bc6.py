import time
import random
import sys
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # Identify Primes
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0:
                return False
        return True

    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    # Large Primes: p > 128
    # Small/Medium Primes: p <= 128
    large_primes = [p for p in primes if p > 128]
    small_primes = [p for p in primes if p <= 128]
    
    # Initialize with Large Primes
    S = set(large_primes)
    
    # Precompute pairwise conflict matrix for all candidates
    # conflict[i][j] = True if i and j cannot both be in a valid set
    # Actually, we need triple constraints, so let's precompute differently.
    # For each candidate x, precompute which pairs (b,c) in the current set would conflict with x.
    
    # Better approach: precompute for each x, the set of values y such that x|y*z or y|x*z or z|x*y
    # This is complex. Let's use a faster can_add with precomputed divisibility.
    
    # Precompute: for each x in 2..256, which y in 2..256 have x | y*z for some z?
    # Actually, let's just optimize the can_add function.
    
    def can_add_fast(x, S_list):
        """Check if x can be added to S_list (assumed valid)."""
        n = len(S_list)
        # Check x | b*c for all pairs b,c in S_list
        for i in range(n):
            b = S_list[i]
            for j in range(i+1, n):
                c = S_list[j]
                if (b * c) % x == 0:
                    return False
                if (x * c) % b == 0:
                    return False
                if (x * b) % c == 0:
                    return False
        return True
    
    # Strategy: Try multiple greedy orderings, then local search with swaps.
    
    best_S = set(large_primes)
    best_size = len(best_S)
    
    # Multiple greedy passes with different orderings
    orderings = [
        'random',
        'ascending', 
        'descending',
        'prime_first',
        'composite_first'
    ]
    
    for ordering in orderings:
        if time.time() > deadline - 10:
            break
        
        current_S = set(large_primes)
        
        # Build candidate list based on ordering
        if ordering == 'random':
            cand_list = composites[:] + small_primes[:]
            random.shuffle(cand_list)
        elif ordering == 'ascending':
            cand_list = sorted(composites) + sorted(small_primes)
        elif ordering == 'descending':
            cand_list = sorted(composites, reverse=True) + sorted(small_primes, reverse=True)
        elif ordering == 'prime_first':
            cand_list = small_primes[:] + composites[:]
            random.shuffle(cand_list)
        elif ordering == 'composite_first':
            cand_list = composites[:] + small_primes[:]
            random.shuffle(cand_list)
        
        S_list = list(current_S)
        for x in cand_list:
            if time.time() > deadline - 10:
                break
            if can_add_fast(x, S_list):
                current_S.add(x)
                S_list.append(x)
        
        if len(current_S) > best_size:
            best_size = len(current_S)
            best_S = set(current_S)
    
    # Local search: try to swap elements to improve
    # For each element in best_S, try removing it and adding a different element
    best_list = sorted(best_S)
    all_candidates = candidates[:]
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        best_list = sorted(best_S)
        n = len(best_list)
        
        for i in range(n):
            if time.time() > deadline - 5:
                break
            # Try removing best_list[i] and adding a different element
            removed = best_list[i]
            temp_S = set(best_list)
            temp_S.remove(removed)
            temp_list = list(temp_S)
            
            # Try adding each candidate not in temp_S
            for x in all_candidates:
                if x in temp_S:
                    continue
                if can_add_fast(x, temp_list):
                    new_S = temp_S | {x}
                    if len(new_S) > best_size:
                        best_S = new_S
                        best_size = len(best_S)
                        improved = True
                        break
                elif len(new_S) == best_size:
                    # Same size, but maybe better for future swaps? Keep for now.
                    pass
    
    # Final: try adding more elements
    best_list = list(best_S)
    for x in all_candidates:
        if time.time() > deadline - 2:
            break
        if x not in best_S:
            if can_add_fast(x, best_list):
                best_S.add(x)
                best_list.append(x)
                best_size = len(best_S)
    
    # Ensure types
    return list(best_S)
