"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt
from functools import lru_cache

N_MAX = 256

def _get_primes(n):
    """Return list of primes up to n."""
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def _can_add(S, x):
    """
    Check if x can be added to set S.
    S is a frozenset of ints for faster hashing.
    """
    s_list = list(S)
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        for j in range(i + 1, n):
            b = s_list[j]
            # Check x | a * b
            if (a * b) % x == 0:
                return False
            # Check a | x * b
            if (x * b) % a == 0:
                return False
            # Check b | x * a
            if (x * a) % b == 0:
                return False
    return True

def _is_valid_set(S):
    """Verify the full 3-wise constraint for set S."""
    s_list = list(S)
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        for j in range(i + 1, n):
            b = s_list[j]
            for k in range(j + 1, n):
                c = s_list[k]
                if (b * c) % a == 0:
                    return False
                if (a * c) % b == 0:
                    return False
                if (a * b) % c == 0:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = N_MAX
    all_candidates = list(range(2, n + 1))
    
    # Strategy: Start with upper half [129, 256] which is an antichain (no element divides another)
    # This gives 128 elements. Then try to add lower elements and do local search.
    
    S = set(range(129, n + 1))
    
    # Try to add elements from lower half in descending order (larger elements are "safer")
    for x in range(128, 1, -1):
        if time.time() > deadline:
            break
        if x not in S:
            if _can_add(frozenset(S), x):
                S.add(x)
    
    # Also try primes and other special elements
    primes = _get_primes(n)
    for p in primes:
        if time.time() > deadline:
            break
        if p not in S:
            if _can_add(frozenset(S), p):
                S.add(p)
    
    best_S = S.copy()
    best_size = len(S)
    
    # Local search with multiple restarts and swaps
    improved = True
    iteration = 0
    while improved and time.time() < deadline:
        improved = False
        iteration += 1
        
        # Try to add any element
        added = False
        for x in all_candidates:
            if time.time() > deadline:
                break
            if x not in S:
                if _can_add(frozenset(S), x):
                    S.add(x)
                    added = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S.copy()
        if added:
            continue
        
        # If no add possible, try swaps
        s_list = list(S)
        c_list = [x for x in all_candidates if x not in S]
        
        if not s_list or not c_list:
            break
        
        # Randomize order for diversity
        random.shuffle(s_list)
        random.shuffle(c_list)
        
        swaps_tried = 0
        max_swaps = 30 + iteration * 5
        
        for y in s_list:
            if swaps_tried >= max_swaps or time.time() > deadline:
                break
            for x in c_list:
                if swaps_tried >= max_swaps or time.time() > deadline:
                    break
                if x == y:
                    continue
                
                S_trial = S - {y}
                if _can_add(frozenset(S_trial), x):
                    S = S_trial | {x}
                    improved = True
                    swaps_tried += 1
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S.copy()
                    break
            if improved:
                break
        
        # Periodically try a different starting point
        if iteration % 20 == 0 and time.time() < deadline:
            # Try starting from a random subset of current best
            if random.random() < 0.3:
                S = set(random.sample(list(best_S), max(1, len(best_S) - 5)))
    
    # Final verification
    if not _is_valid_set(best_S):
        # Fallback: use primes only
        best_S = set(_get_primes(n))
    
    return [int(x) for x in best_S]

# EVOLVE-BLOCK-END
