import time
import itertools
from math import isqrt

def factorize(n):
    """Return set of prime factors of n."""
    if n <= 1:
        return set()
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    # Try small prime sets, increasing in size
    # Start with the most promising: {2, 3, 5, 7}
    
    # Precompute: for a given prime set P, find all numbers <= 10000 
    # such that all prime factors of the number are in P
    # Then check pairwise sums
    
    prime_sets_to_try = [
        {2, 3},
        {2, 3, 5},
        {2, 3, 5, 7},
        {2, 3, 5, 7, 11},
        {2, 3, 5, 7, 11, 13},
    ]
    
    best_result = None
    best_prime_count = float('inf')
    
    for prime_set in prime_sets_to_try:
        if time.time() > deadline:
            break
            
        # Find all numbers <= 10000 whose prime factors are all in prime_set
        valid_numbers = []
        for n in range(1, 10001):
            pf = factorize(n)
            if pf.issubset(prime_set):
                valid_numbers.append(n)
        
        if len(valid_numbers) < 12:
            continue
        
        # Now search for 12 numbers from valid_numbers such that 
        # all pairwise sums also have prime factors in prime_set
        # Use greedy/backtracking
        
        n_valid = len(valid_numbers)
        
        # Precompute pairwise sum validity
        # sum_ok[i][j] = True if prime_factors(valid_numbers[i] + valid_numbers[j]) subset of prime_set
        # This is O(n_valid^2) which could be large
        
        # Limit to a subset for feasibility
        # Try a subset of valid_numbers, maybe the first few hundred
        max_search = min(n_valid, 500)
        search_numbers = valid_numbers[:max_search]
        ns = len(search_numbers)
        
        # Precompute sum validity
        sum_ok = [[False] * ns for _ in range(ns)]
        for i in range(ns):
            for j in range(i + 1, ns):
                s = search_numbers[i] + search_numbers[j]
                pf_s = factorize(s)
                if pf_s.issubset(prime_set):
                    sum_ok[i][j] = True
                    sum_ok[j][i] = True
        
        # Backtracking search for 12 numbers
        chosen = []
        
        def backtrack(start_idx):
            if time.time() > deadline:
                return False
            if len(chosen) == 12:
                return True
            remaining = 12 - len(chosen)
            if start_idx + remaining > ns:
                return False
            
            for i in range(start_idx, ns):
                # Check if search_numbers[i] is compatible with all chosen
                compatible = True
                for c_idx, c_val in enumerate(chosen):
                    # Find index of c_val in search_numbers
                    # This is inefficient; let's store indices
                    pass
                
                # Better: store chosen as indices
                break
            
            return False
        
        # Let me redo with indices
        chosen_idx = []
        
        def backtrack2(start_idx):
            if time.time() > deadline:
                return False
            if len(chosen_idx) == 12:
                return True
            remaining = 12 - len(chosen_idx)
            if start_idx + remaining > ns:
                return False
            
            for i in range(start_idx, ns):
                # Check compatibility with all chosen
                ok = True
                for j in chosen_idx:
                    if j > i:
                        if not sum_ok[i][j]:
                            ok = False
                            break
                    else:
                        if not sum_ok[j][i]:
                            ok = False
                            break
                if ok:
                    chosen_idx.append(i)
                    if backtrack2(i + 1):
                        return True
                    chosen_idx.pop()
            
            return False
        
        if backtrack2(0):
            result = [search_numbers[i] for i in chosen_idx]
            # Verify
            primes_found = set()
            for i in range(12):
                for j in range(i + 1, 12):
                    primes_found |= factorize(result[i] + result[j])
            if len(primes_found) < best_prime_count:
                best_prime_count = len(primes_found)
                best_result = result
            break  # Found solution for this prime set, move to next
    
    if best_result is not None:
        return best_result
    
    # Fallback: return the seed
    return list(range(1, 13))
