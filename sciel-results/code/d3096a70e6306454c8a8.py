"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

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

def _precompute_conflict_step(n):
    """Precompute conflict_step[a][x] = a // gcd(a, x) for a,x in 2..n.
    Then (x*y) % a == 0 iff y % conflict_step[a][x] == 0."""
    conflict_step = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(2, n + 1):
        for x in range(2, n + 1):
            g = gcd(a, x)
            conflict_step[a][x] = a // g
    return conflict_step

def _can_add_fast(candidate, current_list, conflict_step):
    """Check if candidate can be added to current_list without violating the constraint."""
    n = len(current_list)
    
    # Case 1: candidate plays role of 'a': candidate ∤ b*c for all distinct b,c in current_list
    for i in range(n):
        b = current_list[i]
        for j in range(i + 1, n):
            c = current_list[j]
            if (b * c) % candidate == 0:
                return False
    
    # Case 2: candidate plays role of 'b' (or 'c'): for each a in current_list,
    #         a ∤ candidate * c for all distinct c in current_list (c != a)
    for i in range(n):
        a = current_list[i]
        step = conflict_step[a][candidate]
        if step == 1:
            # All c satisfy (candidate*c) % a == 0
            if n >= 2:
                return False
        else:
            for j in range(n):
                if j == i:
                    continue
                c = current_list[j]
                if c % step == 0:
                    return False
    return True

def _greedy_with_ordering(ordering, conflict_step):
    """Greedy construction with a specific ordering of composites."""
    primes = [a for a in range(2, P['n']+1) if _is_prime(a)]
    current = list(primes)
    current_set = set(primes)
    
    for cand in ordering:
        if _can_add_fast(cand, current, conflict_step):
            current.append(cand)
            current_set.add(cand)
    return current_set

def _local_search_1for1(current_set, conflict_step, deadline):
    """1-for-1 swap local search."""
    best_set = set(current_set)
    best_size = len(best_set)
    all_elements = list(range(2, P['n']+1))
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_list = list(best_set)
        
        for u in current_list:
            if time.time() > deadline - 5:
                break
            trial_set = best_set - {u}
            trial_list = [x for x in current_list if x != u]
            
            for v in all_elements:
                if v in trial_set:
                    continue
                if _can_add_fast(v, trial_list, conflict_step):
                    new_set = trial_set | {v}
                    new_size = len(new_set)
                    if new_size > best_size:
                        best_set = new_set
                        best_size = new_size
                        improved = True
                        break
        # After finding an improvement, restart the outer loop
    
    return best_set

def construct():
    deadline = time.time() + 50  # safety margin
    
    # Precompute conflict steps
    conflict_step = _precompute_conflict_step(P['n'])
    
    # Phase 1: Greedy with natural ordering
    composites = [a for a in range(2, P['n']+1) if not _is_prime(a)]
    greedy_set = _greedy_with_ordering(composites, conflict_step)
    
    best_result = greedy_set
    best_size = len(greedy_set)
    
    # Phase 2: Multiple random restarts
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        ordering = list(composites)
        random.shuffle(ordering)
        current_set = _greedy_with_ordering(ordering, conflict_step)
        
        if len(current_set) > best_size:
            best_result = current_set
            best_size = len(current_set)
        
        # Local search
        if time.time() < deadline - 10:
            improved_set = _local_search_1for1(current_set, conflict_step, deadline)
            if len(improved_set) > best_size:
                best_result = improved_set
                best_size = len(improved_set)
    
    # Final local search on best result
    if time.time() < deadline - 5:
        final_set = _local_search_1for1(best_result, conflict_step, deadline)
        if len(final_set) > best_size:
            best_result = final_set
            best_size = len(final_set)
    
    return [int(x) for x in sorted(best_result)]
# EVOLVE-BLOCK-END
