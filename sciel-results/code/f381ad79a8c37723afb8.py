import time
import sys
from math import isqrt

def _sieve_smallest_prime(n):
    """Return array sp where sp[i] is smallest prime factor of i."""
    sp = list(range(n + 1))
    for i in range(2, isqrt(n) + 1):
        if sp[i] == i:  # prime
            for j in range(i * i, n + 1, i):
                if sp[j] == j:
                    sp[j] = i
    return sp

def _factorize(n, sp):
    """Return set of prime factors of n using smallest prime factor table."""
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = sp[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _is_smooth(n, allowed_primes_set, sp):
    """Check if n is smooth over allowed_primes_set."""
    if n <= 1:
        return True
    while n > 1:
        p = sp[n]
        if p not in allowed_primes_set:
            return False
        while n % p == 0:
            n //= p
    return True

def _generate_smooth_numbers(max_val, primes, sp):
    """Generate all numbers <= max_val whose prime factors are in primes."""
    primes_set = set(primes)
    result = set()
    
    def gen(idx, current):
        if current > max_val:
            return
        if current > 1:
            result.add(current)
        if idx >= len(primes):
            return
        p = primes[idx]
        val = current
        while val <= max_val:
            result.add(val)
            if val > max_val // p:
                break
            val *= p
            gen(idx + 1, val)
    
    gen(0, 1)
    return sorted(result)

def _find_clique(compat, target_size, deadline):
    """Find a clique of target_size in graph defined by compat (list of sets)."""
    n = len(compat)
    if n < target_size:
        return None
    
    # Order nodes by degree (descending) for better pruning
    degrees = [len(compat[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    # Map to reordered indices
    # Use original indices but process in order
    best_clique = []
    
    def backtrack(candidates, clique):
        if time.time() > deadline:
            return
        if len(clique) == target_size:
            best_clique.extend(clique)
            return
        if len(candidates) < target_size - len(clique):
            return
        
        # Try each candidate
        for i in range(len(candidates)):
            node = candidates[i]
            new_clique = clique + [node]
            # Filter candidates to those compatible with node and after i
            new_candidates = []
            for j in range(i + 1, len(candidates)):
                other = candidates[j]
                if other in compat[node]:
                    new_candidates.append(other)
            
            if len(new_candidates) >= target_size - len(new_clique):
                backtrack(new_candidates, new_clique)
                if best_clique:
                    return
    
    backtrack(order, [])
    return best_clique if best_clique else None

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin
    
    MAX_VAL = 20000  # max sum of two numbers <= 10000
    N = 12
    MAX_NUM = 10000
    
    # Precompute smallest prime factor table
    sp = _sieve_smallest_prime(MAX_VAL)
    
    # Trivial solution as fallback
    trivial = list(range(1, 13))
    trivial_primes = set()
    for i in range(N):
        for j in range(i + 1, N):
            s = trivial[i] + trivial[j]
            trivial_primes |= _factorize(s, sp)
    best_solution = trivial
    best_prime_count = len(trivial_primes)
    
    # Candidate prime sets to try, in order of increasing size
    # Start with small sets
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Try prime sets of increasing size
    # For each size, try combinations
    found_solution = None
    
    # First, try very small sets
    for size in range(2, min(8, len(small_primes) + 1)):
        if time.time() > deadline - 5:
            break
        
        # Try combinations of this size
        # For efficiency, try sets containing small primes first
        from itertools import combinations
        
        tried = 0
        for combo in combinations(small_primes, size):
            if time.time() > deadline - 5:
                break
            tried += 1
            
            primes_set = set(combo)
            
            # Generate smooth numbers <= MAX_VAL
            smooth = _generate_smooth_numbers(MAX_VAL, list(combo), sp)
            
            # Filter to numbers <= MAX_NUM
            candidates = [x for x in smooth if 1 <= x <= MAX_NUM]
            
            if len(candidates) < N:
                continue
            
            # Build compatibility: two numbers a,b compatible if a+b is smooth
            # For efficiency, limit candidates
            if len(candidates) > 500:
                # Take a subset - maybe prefer smaller numbers or random
                import random
                random.seed(42)
                candidates = random.sample(candidates, 500)
            
            # Build compatibility graph
            compat = [set() for _ in range(len(candidates))]
            for i in range(len(candidates)):
                ai = candidates[i]
                for j in range(i + 1, len(candidates)):
                    aj = candidates[j]
                    s = ai + aj
                    if _is_smooth(s, primes_set, sp):
                        compat[i].add(j)
                        compat[j].add(i)
            
            # Find clique of size N
            clique = _find_clique(compat, N, deadline - 2)
            if clique:
                solution = [candidates[i] for i in clique]
                # Verify
                primes_used = set()
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        s = solution[i] + solution[j]
                        pf = _factorize(s, sp)
                        primes_used |= pf
                if len(primes_used) < best_prime_count:
                    best_solution = solution
                    best_prime_count = len(primes_used)
                    found_solution = solution
                    break  # Move to next (larger) prime set size? No, we want smallest
                # Actually we found a solution with this prime set, but let's continue
                # to see if we can do better with same or smaller set
                break  # This combo works, try next combo of same size
        
        if found_solution:
            break  # Found with this prime set size, try smaller? No, we went increasing
    
    # If we haven't found a good solution, try a different approach:
    # Search for numbers where pairwise sums have few distinct primes
    if time.time() < deadline - 10:
        # Try to find 12 numbers with sums using only {2,3,5}
        # These are 5-smooth numbers
        smooth_235 = _generate_smooth_numbers(MAX_VAL, [2, 3, 5], sp)
        candidates = [x for x in smooth_235 if 1 <= x <= MAX_NUM]
        
        if len(candidates) >= N:
            if len(candidates) > 300:
                import random
                random.seed(123)
                candidates = random.sample(candidates, 300)
            
            compat = [set() for _ in range(len(candidates))]
            for i in range(len(candidates)):
                ai = candidates[i]
                for j in range(i + 1, len(candidates)):
                    aj = candidates[j]
                    s = ai + aj
                    if _is_smooth(s, {2, 3, 5}, sp):
                        compat[i].add(j)
                        compat[j].add(i)
            
            clique = _find_clique(compat, N, deadline - 2)
            if clique:
                solution = [candidates[i] for i in clique]
                primes_used = set()
                for i in range(N):
                    for j in range(i + 1, N):
                        s = solution[i] + solution[j]
                        primes_used |= _factorize(s, sp)
                if len(primes_used) < best_prime_count:
                    best_solution = solution
                    best_prime_count = len(primes_used)
    
    return best_solution
