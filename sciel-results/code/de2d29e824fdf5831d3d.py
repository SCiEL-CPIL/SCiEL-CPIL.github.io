"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

P = {'n': 256}

def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            return False
    return True

def _find_violation(s):
    """
    Find a violating triple (a, b, c) in set s.
    Returns (a, b, c) such that a divides b*c, or None if valid.
    """
    lst = list(s)
    # Optimization: sort by value? 
    # If a is small, it's more likely to divide b*c.
    # So iterate a in ascending order.
    for i in range(len(lst)):
        a = lst[i]
        for j in range(i+1, len(lst)):
            b = lst[j]
            for k in range(j+1, len(lst)):
                c = lst[k]
                if (b * c) % a == 0:
                    return (a, b, c)
    return None

def _valid(s):
    return _find_violation(s) is None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = P['n']
    candidates = list(range(2, n + 1))
    
    primes = [p for p in candidates if _is_prime(p)]
    composites = [p for p in candidates if not _is_prime(p)]
    
    # Initial set: All primes
    current_set = set(primes)
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Pre-compute factors or just rely on modulo
    # To speed up, we can check if a > b*c? No, b*c is large.
    
    # Local Search
    # We will try to add composites.
    # If a composite x conflicts, we remove one element from the conflict.
    
    # Shuffle composites to avoid bias
    random.shuffle(composites)
    
    # We also allow swapping primes with composites if it helps?
    # Actually, just adding and removing is enough.
    
    # To ensure we explore, we will iterate through composites multiple times.
    # And we will also try removing primes to add composites.
    
    all_candidates = candidates[:]
    random.shuffle(all_candidates)
    
    step = 0
    while time.time() < deadline:
        # Pick a candidate to add
        # Strategy: Randomly pick from all candidates not in current_set
        available = [x for x in all_candidates if x not in current_set]
        if not available:
            break
            
        x = random.choice(available)
        
        trial_set = current_set | {x}
        
        violation = _find_violation(trial_set)
        
        if violation is None:
            # Valid, accept
            current_set = trial_set
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
        else:
            # Invalid. Remove one element from the violating triple to make room for x?
            # Or just reject x?
            # If we reject, we might get stuck.
            # Let's try to resolve the conflict by removing one of a, b, c.
            a, b, c = violation
            
            # Heuristic: Remove the element that is "most composite" or "smallest"?
            # Actually, removing a prime is bad. Removing a composite is better.
            # Let's try removing each of a, b, c and see if the set becomes valid with x added.
            # Note: Removing one element might not fix all conflicts if there are multiple.
            # But _find_violation returns the first one.
            
            # Try removing a
            trial_remove_a = trial_set - {a}
            if _valid(trial_remove_a):
                current_set = trial_remove_a
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                continue
            
            # Try removing b
            trial_remove_b = trial_set - {b}
            if _valid(trial_remove_b):
                current_set = trial_remove_b
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                continue
            
            # Try removing c
            trial_remove_c = trial_set - {c}
            if _valid(trial_remove_c):
                current_set = trial_remove_c
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                continue
            
            # If none of the removals fix it immediately (or if we want to be more aggressive)
            # We could try removing a random element from the set, not just the triple.
            # But that's expensive.
            # Let's just reject x for now.
            pass
        
        step += 1
        if step % 1000 == 0:
             # Periodic restart or perturbation?
             # If stuck, maybe remove a random prime and try to add composites?
             # For now, keep it simple.
             pass

    result = sorted(best_set)
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
