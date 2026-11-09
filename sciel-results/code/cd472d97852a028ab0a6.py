"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

P = {'n': 256}

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

_PRIMES = [a for a in range(2, P['n'] + 1) if _is_prime(a)]
_COMPOSITES = [a for a in range(2, P['n'] + 1) if not _is_prime(a)]
_ALL_NUMS = list(range(2, P['n'] + 1))

def _can_add_fast(candidate, current_list):
    """Check if candidate can be added to current_list without violating the constraint."""
    n = len(current_list)
    
    # Case 1: candidate plays role of 'a': candidate ∤ b*c for all distinct b,c in current_list
    for i in range(n):
        for j in range(i + 1, n):
            if (current_list[i] * current_list[j]) % candidate == 0:
                return False
    
    # Case 2: candidate plays role of 'b' (or 'c'): for each a in current_list, a ∤ candidate * b
    for i in range(n):
        a = current_list[i]
        for j in range(n):
            if j == i:
                continue
            b = current_list[j]
            if (candidate * b) % a == 0:
                return False
    return True

def _validate_set(s):
    """Verify that a set satisfies the constraint for all triples."""
    elems = sorted(s)
    n = len(elems)
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                if (elems[j] * elems[k]) % elems[i] == 0:
                    return False
    return True

def _greedy_construct(order='increasing', rng=None):
    """Greedy construction starting from primes."""
    current = set(_PRIMES)
    current_list = sorted(current)
    
    composites = _COMPOSITES[:]
    if order == 'decreasing':
        composites.reverse()
    elif order == 'random':
        if rng is not None:
            rng.shuffle(composites)
        else:
            random.shuffle(composites)
    
    for cand in composites:
        if _can_add_fast(cand, current_list):
            current.add(cand)
            current_list.append(cand)
            current_list.sort()
    return current

def _greedy_refill(trial_set, candidates, rng):
    """Greedily add candidates to trial_set in the given order."""
    current = set(trial_set)
    current_list = sorted(current)
    for cand in candidates:
        if cand in current:
            continue
        if _can_add_fast(cand, current_list):
            current.add(cand)
            current_list.append(cand)
            current_list.sort()
    return current

def _local_search(initial_set, deadline):
    """Local search with 1-swap, 2-swap, and random perturbation moves."""
    best = set(initial_set)
    best_list = sorted(best)
    best_size = len(best)
    rng = random.Random(12345)
    
    while time.time() < deadline - 5:
        improved = False
        
        # Try 1-swap: remove one element, try to add a better one
        for idx in range(len(best_list)):
            if time.time() > deadline - 5:
                break
            elem = best_list[idx]
            trial = best - {elem}
            trial_list = best_list[:idx] + best_list[idx+1:]
            
            for cand in range(2, P['n'] + 1):
                if cand in trial:
                    continue
                if _can_add_fast(cand, trial_list):
                    new_set = trial | {cand}
                    if len(new_set) > best_size:
                        best = new_set
                        best_list = sorted(best)
                        best_size = len(best)
                        improved = True
                        break
        
        if not improved:
            # Try 2-swap: remove two elements, try to add two better ones
            found = False
            for i in range(len(best_list)):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, len(best_list)):
                    if time.time() > deadline - 5:
                        break
                    trial = best - {best_list[i]} - {best_list[j]}
                    trial_list = [x for x in best_list if x != best_list[i] and x != best_list[j]]
                    
                    # Try adding two candidates
                    for cand1 in range(2, P['n'] + 1):
                        if cand1 in trial:
                            continue
                        if not _can_add_fast(cand1, trial_list):
                            continue
                        trial2_list = trial_list + [cand1]
                        trial2 = trial | {cand1}
                        
                        for cand2 in range(2, P['n'] + 1):
                            if cand2 in trial2 or cand2 == cand1:
                                continue
                            if _can_add_fast(cand2, trial2_list):
                                new_set = trial2 | {cand2}
                                if len(new_set) > best_size:
                                    best = new_set
                                    best_list = sorted(best)
                                    best_size = len(new_set)
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                improved = True
        
        if not improved:
            # Random perturbation: remove 2-3 elements, greedily refill
            perturbed = False
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                num_remove = rng.choice([2, 3])
                if num_remove >= len(best_list):
                    continue
                removed = rng.sample(best_list, num_remove)
                trial = best - set(removed)
                trial2 = _greedy_refill(trial, _ALL_NUMS, rng)
                if len(trial2) > best_size:
                    best = trial2
                    best_list = sorted(best)
                    best_size = len(trial2)
                    improved = True
                    perturbed = True
                    break
            
            if not improved and not perturbed:
                break
    
    return best

def construct():
    deadline = time.time() + 55  # safety margin before 90s kill
    
    # Phase 1: Multiple greedy constructions with different orderings
    candidates = []
    
    # Increasing order
    g1 = _greedy_construct('increasing')
    candidates.append(g1)
    
    # Decreasing order  
    g2 = _greedy_construct('decreasing')
    candidates.append(g2)
    
    # Random orders with different seeds
    for seed_val in range(15):
        rng = random.Random(seed_val)
        g = _greedy_construct('random', rng=rng)
        candidates.append(g)
    
    # Pick the best greedy result
    best_set = max(candidates, key=len)
    
    # Phase 2: Local search improvement
    result = _local_search(best_set, deadline)
    
    # Verify the result
    if not _validate_set(result):
        # Fallback to just primes if validation fails
        result = set(_PRIMES)
    
    return [int(x) for x in sorted(result)]
# EVOLVE-BLOCK-END
