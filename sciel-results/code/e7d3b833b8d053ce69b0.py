"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

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

def _can_add(S, x):
    """Check if x can be added to set S without violating constraint.
    For all distinct a,b in S: a ∤ b*x, b ∤ a*x, x ∤ a*b
    """
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if i == j:
                continue
            b = S_list[j]
            if (b * x) % a == 0:
                return False
            if (a * x) % b == 0:
                return False
            if (a * b) % x == 0:
                return False
    return True

def _is_valid(S):
    """Check if set S is valid: for all distinct a,b,c in S, a ∤ b*c"""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if i == j:
                continue
            b = S_list[j]
            for k in range(n):
                if i == k or j == k:
                    continue
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    N = 256
    all_nums = list(range(2, N + 1))
    primes = [x for x in all_nums if _is_prime(x)]
    composites = [x for x in all_nums if not _is_prime(x)]
    
    # Start with all primes
    S = set(primes)
    best = list(S)
    best_size = len(best)
    
    # Greedy: try adding composites in random order
    random.seed(42)
    composites_shuffled = composites[:]
    random.shuffle(composites_shuffled)
    for x in composites_shuffled:
        if time.time() > deadline:
            break
        if _can_add(S, x):
            S.add(x)
    
    if len(S) > best_size:
        best = list(S)
        best_size = len(S)
    
    # Local search: try to improve
    S_list = list(S)
    random.shuffle(S_list)
    
    for attempt in range(50000):
        if time.time() > deadline:
            break
        
        # Try adding any composite not in S
        added = False
        for x in composites_shuffled:
            if x in S:
                continue
            if _can_add(S, x):
                S.add(x)
                S_list.append(x)
                added = True
                break
        
        if not added:
            # Try a swap: remove a random element, try to add composites
            if S_list:
                remove_idx = random.randint(0, len(S_list) - 1)
                removed = S_list[remove_idx]
                S.discard(removed)
                S_list.pop(remove_idx)
                
                added_swap = False
                for x in composites_shuffled:
                    if x in S:
                        continue
                    if _can_add(S, x):
                        S.add(x)
                        S_list.append(x)
                        added_swap = True
                        break
                
                if not added_swap:
                    # Put it back
                    S.add(removed)
                    S_list.append(removed)
    
    if len(S) > best_size:
        best = list(S)
        best_size = len(S)
    
    # Verify and return
    result = sorted(best)
    if not _is_valid(result):
        result = sorted(primes)
    
    return result
# EVOLVE-BLOCK-END
