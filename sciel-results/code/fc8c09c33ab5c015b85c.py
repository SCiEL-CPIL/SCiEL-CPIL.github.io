"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

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

def _num_prime_factors(n):
    """Count distinct prime factors of n."""
    count = 0
    d = 2
    while d * d <= n:
        if n % d == 0:
            count += 1
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def _can_add_fast(candidate, current_list):
    """Fast check: can candidate be added to current_list (a list of ints)?
    Returns True if for all distinct a,b,c in current_list ∪ {candidate}, a ∤ b*c."""
    n = len(current_list)
    
    # Case 1: candidate as 'a': candidate ∤ b*c for all distinct b,c in current_list
    for i in range(n):
        for j in range(i+1, n):
            if (current_list[i] * current_list[j]) % candidate == 0:
                return False
    
    # Case 2: candidate as 'b' or 'c': for each a in current_list, a ∤ candidate * b
    # where b is another distinct element from current_list (b != a, b != candidate)
    for i in range(n):
        a = current_list[i]
        for j in range(n):
            if j == i:
                continue
            b = current_list[j]
            if (candidate * b) % a == 0:
                return False
    return True

def _greedy_construct():
    """Greedy construction starting from primes, adding composites by fewest prime factors first."""
    primes = [a for a in range(2, P['n']+1) if _is_prime(a)]
    current = set(primes)
    current_list = sorted(current)
    
    # Sort composites by number of distinct prime factors (ascending), then by value
    composites = [a for a in range(2, P['n']+1) if not _is_prime(a)]
    composites.sort(key=lambda x: (_num_prime_factors(x), x))
    
    for cand in composites:
        if _can_add_fast(cand, current_list):
            current.add(cand)
            current_list.append(cand)
            current_list.sort()
    return current

def _local_search(initial_set, deadline):
    """Enhanced local search with 1-swap moves and multiple passes."""
    best = set(initial_set)
    best_list = sorted(best)
    improved = True
    pass_count = 0
    
    while improved and time.time() < deadline - 5:
        improved = False
        pass_count += 1
        best_list = sorted(best)
        n = len(best_list)
        
        for i in range(n):
            if time.time() > deadline - 5:
                return best
            elem = best_list[i]
            trial_set = best - {elem}
            trial_list = [x for x in best_list if x != elem]
            
            # Try all candidates not in trial_set, ordered by fewest prime factors
            candidates = [c for c in range(2, P['n']+1) if c not in trial_set]
            candidates.sort(key=lambda x: (_num_prime_factors(x), x))
            
            for cand in candidates:
                if _can_add_fast(cand, trial_list):
                    new_set = trial_set | {cand}
                    if len(new_set) > len(best):
                        best = new_set
                        improved = True
                        break
            if improved:
                break
    return best

def _verify_set(s):
    """Verify that the set satisfies the constraint."""
    lst = sorted(s)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                b = lst[j]
                c = lst[k]
                if (b * c) % a == 0:
                    return False
    return True

def construct():
    deadline = time.time() + 55  # safety margin before 60s limit
    
    # Phase 1: Greedy construction
    greedy_set = _greedy_construct()
    
    # Phase 2: Local search improvement
    result = _local_search(greedy_set, deadline)
    
    # Phase 3: Verify and if invalid, fall back to primes only
    if not _verify_set(result):
        primes = [a for a in range(2, P['n']+1) if _is_prime(a)]
        result = set(primes)
    
    return [int(x) for x in sorted(result)]
# EVOLVE-BLOCK-END
