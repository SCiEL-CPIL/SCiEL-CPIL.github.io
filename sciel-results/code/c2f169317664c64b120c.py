"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def _factorize(n):
    """Return prime factorization as tuple of (prime, exponent) pairs."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            cnt = 0
            while n % d == 0:
                n //= d
                cnt += 1
            factors.append((d, cnt))
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append((n, 1))
    return tuple(factors)

def _exponent_vector(n, primes):
    """Return the exponent vector of n with respect to given primes."""
    vec = []
    temp = n
    for p in primes:
        cnt = 0
        while temp % p == 0:
            temp //= p
            cnt += 1
        vec.append(cnt)
    if temp != 1:
        # This shouldn't happen if primes include all primes up to 40
        vec.append(temp)  # fallback
    return tuple(vec)

def _subset_products_distinct(nums):
    """Check if all 2^k subset products of nums are distinct."""
    k = len(nums)
    products = set()
    # Use iterative construction: for each element, double the set of products
    # Start with {1} (empty product)
    prods = {1}
    for num in nums:
        new_prods = set()
        for p in prods:
            np = p * num
            if np in prods or np in new_prods:
                return False
            new_prods.add(np)
        prods |= new_prods
    return len(prods) == (1 << k)

def _greedy_with_collision_check(candidates, target_size, deadline):
    """Greedy approach: try to add elements one by one, checking for collisions."""
    selected = []
    prods = {1}
    
    # Sort candidates by some heuristic - maybe by number of prime factors or just value
    for num in candidates:
        if time.time() > deadline:
            break
        if len(selected) >= target_size:
            break
        # Check if adding num creates any collision
        new_prods = set()
        valid = True
        for p in prods:
            np = p * num
            if np in prods or np in new_prods:
                valid = False
                break
            new_prods.add(np)
        if valid:
            selected.append(num)
            prods |= new_prods
    
    return selected

def _backtrack_search(candidates, target_size, deadline, current_prods=None, current_selected=None):
    """Backtracking search to find a subset of target_size with distinct subset products."""
    if current_prods is None:
        current_prods = {1}
    if current_selected is None:
        current_selected = []
    
    if len(current_selected) == target_size:
        return current_selected
    
    remaining = target_size - len(current_selected)
    
    # If not enough candidates left, fail
    if len(candidates) < remaining:
        return None
    
    # Try each candidate
    for i, num in enumerate(candidates):
        if time.time() > deadline:
            return None
        
        # Check if adding num creates any collision
        new_prods = set()
        valid = True
        for p in current_prods:
            np = p * num
            if np in current_prods or np in new_prods:
                valid = False
                break
            new_prods.add(np)
        
        if not valid:
            continue
        
        # Recurse
        result = _backtrack_search(candidates[i+1:], target_size, deadline, 
                                    current_prods | new_prods, current_selected + [num])
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline to be safe
    
    # All integers from 2 to 40
    all_nums = list(range(2, 41))
    
    # Try to find a set of size 18
    # First, use greedy to get a good initial set
    greedy_result = _greedy_with_collision_check(all_nums, 18, deadline)
    
    if len(greedy_result) == 18:
        return list(greedy_result)
    
    # If greedy didn't reach 18, try backtracking with different orderings
    # Try sorting by different criteria
    orderings = [
        all_nums,
        sorted(all_nums, reverse=True),
        sorted(all_nums, key=lambda x: sum(1 for d in range(2, x+1) if x % d == 0)),
        sorted(all_nums, key=lambda x: _factorize(x)[0][1] if _factorize(x) else 0),
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        result = _backtrack_search(ordering, 18, deadline)
        if result is not None and len(result) == 18:
            return list(result)
    
    # If we couldn't find 18, return the best we found
    if len(greedy_result) > 0:
        return list(greedy_result)
    
    # Fallback: just return primes
    from math import isqrt
    primes = [a for a in range(2, 41) if all(a % d for d in range(2, isqrt(a) + 1))]
    return primes
# EVOLVE-BLOCK-END
