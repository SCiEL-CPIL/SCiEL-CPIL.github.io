"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

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
    S is a set of ints.
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

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = N_MAX
    primes = _get_primes(n)
    S = set(primes)
    
    # Greedy Add
    for x in range(2, n + 1):
        if time.time() > deadline:
            break
        if x not in S:
            if _can_add(S, x):
                S.add(x)
    
    # Local Search: Swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add
        added = False
        for x in range(2, n + 1):
            if time.time() > deadline:
                break
            if x not in S:
                if _can_add(S, x):
                    S.add(x)
                    added = True
        if added:
            continue
        
        # If no add, try swaps to change structure
        s_list = list(S)
        c_list = [x for x in range(2, n + 1) if x not in S]
        
        if not s_list or not c_list:
            break
            
        random.shuffle(s_list)
        random.shuffle(c_list)
        
        swaps_tried = 0
        max_swaps = 50
        
        for y in s_list:
            if swaps_tried >= max_swaps or time.time() > deadline:
                break
            for x in c_list:
                if swaps_tried >= max_swaps or time.time() > deadline:
                    break
                if x == y:
                    continue
                
                S_trial = S - {y}
                if _can_add(S_trial, x):
                    S = S_trial | {x}
                    improved = True
                    swaps_tried += 1
                    break
            if improved:
                break

    return [int(x) for x in S]

# EVOLVE-BLOCK-END
