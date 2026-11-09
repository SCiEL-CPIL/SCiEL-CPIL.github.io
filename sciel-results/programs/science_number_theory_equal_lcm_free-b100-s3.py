"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    """Check if {a, b, c} forms a bad triple (equal pairwise LCMs)."""
    return _lcm(a, b) == _lcm(a, c) == _lcm(b, c)

def _conflicts_with_set(x, S):
    """Check if adding x to S creates any bad triple."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            if _is_bad_triple(x, a, b):
                return True
    return False

def _greedy_with_order(order):
    """Add elements from order to a new set if they don't conflict."""
    S = set()
    for x in order:
        if not _conflicts_with_set(x, S):
            S.add(x)
    return S

def _local_search(S, deadline):
    """
    Try to improve S by:
    1. Adding any element that doesn't conflict
    2. Swapping: remove one element, try to add a different one
    Returns the improved set.
    """
    S = set(S)
    full_set = set(range(1, 129))
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any element not in S
        for x in range(1, 129):
            if x in S:
                continue
            if not _conflicts_with_set(x, S):
                S.add(x)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try 1-swaps (remove one, add one)
        found_swap = False
        for a in list(S):
            S.remove(a)
            for x in range(1, 129):
                if x in S:
                    continue
                if not _conflicts_with_set(x, S):
                    S.add(x)
                    found_swap = True
                    break
            if not found_swap:
                S.add(a)
        
        if not found_swap:
            break
    
    return S

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _num_divisors(n):
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
        i += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    all_elements = list(range(1, 129))
    
    # Build several deterministic orderings
    orderings = []
    
    # 1. Primes first, then non-primes
    primes = [x for x in all_elements if _is_prime(x)]
    non_primes = [x for x in all_elements if not _is_prime(x)]
    orderings.append(primes + non_primes)
    
    # 2. By number of divisors (fewer first)
    orderings.append(sorted(all_elements, key=lambda x: _num_divisors(x)))
    
    # 3. Reverse order (largest first)
    orderings.append(list(reversed(all_elements)))
    
    # 4. Forward order (smallest first) - baseline greedy
    orderings.append(all_elements)
    
    # 5-14. Random orderings
    random.seed(12345)
    for _ in range(10):
        orderings.append(random.sample(all_elements, 128))
    
    best_set = set()
    best_size = 0
    
    # Run all deterministic and random greedy passes
    for order in orderings:
        if time.time() > deadline:
            break
        S = _greedy_with_order(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Continue with more random restarts if time permits
    while time.time() < deadline:
        candidates = random.sample(all_elements, 128)
        S = _greedy_with_order(candidates)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Local search on the best set found
    if time.time() < deadline:
        improved = _local_search(best_set, deadline)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Final validation: ensure no bad triples exist
    S_list = sorted(best_set)
    valid = True
    for a, b, c in combinations(S_list, 3):
        if _is_bad_triple(a, b, c):
            valid = False
            break
    
    if not valid:
        # Fallback: use the simplest valid set
        best_set = {1}
    
    if not best_set:
        best_set = {1}
    
    return list(best_set)
# EVOLVE-BLOCK-END
