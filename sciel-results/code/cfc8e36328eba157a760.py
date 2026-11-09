"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

P = {'n': 256}

def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True

def _prime_factors(n):
    """Return a dictionary of prime factors and their exponents."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _a_divides_bc(a, b, c, pf_cache):
    """Check if a divides b*c using precomputed prime factors."""
    fa = pf_cache[a]
    fb = pf_cache[b]
    fc = pf_cache[c]
    for p, exp_a in fa.items():
        exp_bc = fb.get(p, 0) + fc.get(p, 0)
        if exp_bc < exp_a:
            return False
    return True

def _count_violations(s, pf_cache):
    """Count total violations in set s."""
    lst = list(s)
    n = len(lst)
    violations = 0
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            for k in range(j+1, n):
                c = lst[k]
                if a <= b * c and _a_divides_bc(a, b, c, pf_cache):
                    violations += 1
    return violations

def _calc_new_violations(s, x, pf_cache):
    """Count violations introduced by adding x to s."""
    count = 0
    lst = list(s)
    n = len(lst)
    for i in range(n):
        b = lst[i]
        for j in range(n):
            if i == j:
                continue
            c = lst[j]
            if _a_divides_bc(x, b, c, pf_cache):
                count += 1
            if _a_divides_bc(b, x, c, pf_cache):
                count += 1
            if _a_divides_bc(c, x, b, pf_cache):
                count += 1
    return count

def _greedy_add(s, candidates, pf_cache, deadline):
    """Greedily add numbers that don't create violations."""
    current = set(s)
    for x in candidates:
        if x in current:
            continue
        if time.time() > deadline:
            break
        if _calc_new_violations(current, x, pf_cache) == 0:
            current.add(x)
    return current

def _local_search(s, candidates, pf_cache, deadline):
    """Local search with swaps to improve set size while maintaining validity."""
    current = set(s)
    best = set(s)
    best_size = len(best)
    
    # Build list of candidates not in current
    not_in_current = [x for x in candidates if x not in current]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try 1-for-1 swaps
        for remove in list(current):
            if time.time() > deadline:
                break
            temp = current - {remove}
            for add in not_in_current:
                if time.time() > deadline:
                    break
                if add in current:
                    continue
                new_set = temp | {add}
                if len(new_set) <= len(current):
                    continue
                if _count_violations(new_set, pf_cache) == 0:
                    current = new_set
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(best)
                    improved = True
                    break
            if improved:
                break
        # Try 1-for-2 swaps
        if not improved:
            for remove in list(current):
                if time.time() > deadline:
                    break
                temp = current - {remove}
                for add1 in not_in_current:
                    if time.time() > deadline:
                        break
                    if add1 in current:
                        continue
                    for add2 in not_in_current:
                        if time.time() > deadline:
                            break
                        if add2 in current or add2 == add1:
                            continue
                        new_set = temp | {add1, add2}
                        if len(new_set) <= len(current):
                            continue
                        if _count_violations(new_set, pf_cache) == 0:
                            current = new_set
                            if len(current) > best_size:
                                best = set(current)
                                best_size = len(best)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = P['n']
    candidates = list(range(2, n + 1))
    
    # Precompute prime factorizations
    pf_cache = {}
    for x in candidates:
        pf_cache[x] = _prime_factors(x)
    
    # Start with all primes
    primes = [p for p in candidates if _is_prime(p)]
    current_set = set(primes)
    
    # Greedily add composites
    composites = [x for x in candidates if not _is_prime(x)]
    current_set = _greedy_add(current_set, composites, pf_cache, deadline)
    
    best_set = set(current_set)
    best_size = len(best_set)
    
    # Local search with multiple restarts
    random.seed(42)
    
    # First local search from greedy result
    best_set = _local_search(best_set, candidates, pf_cache, deadline)
    best_size = len(best_set)
    
    # Try different initial seeds for local search
    for restart in range(10):
        if time.time() > deadline - 5:
            break
        
        # Create a perturbed initial set
        if restart == 0:
            initial = set(primes)
        else:
            # Start with primes plus some random composites
            initial = set(primes)
            num_composites = min(len(composites), restart * 3)
            random.shuffle(composites)
            for c in composites[:num_composites]:
                if _calc_new_violations(initial, c, pf_cache) == 0:
                    initial.add(c)
        
        # Local search
        result = _local_search(initial, candidates, pf_cache, deadline)
        if len(result) > best_size:
            best_set = result
            best_size = len(best_set)
    
    # Final verification
    if _count_violations(best_set, pf_cache) != 0:
        # Fallback to primes
        best_set = set(primes)
    
    return [int(x) for x in sorted(best_set)]
# EVOLVE-BLOCK-END
