"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

P = {'n': 256}
_DEADLINE = None

def _is_prime(x):
    if x < 2:
        return False
    if x < 4:
        return True
    if x % 2 == 0 or x % 3 == 0:
        return False
    d = 5
    while d * d <= x:
        if x % d == 0 or x % (d + 2) == 0:
            return False
        d += 6
    return True

def _valid_set(S):
    """Check if set S is valid: for all distinct a,b,c in S, a ∤ b*c."""
    S_list = list(S)
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i + 1, m):
            b = S_list[j]
            for k in range(j + 1, m):
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _can_add(S, x):
    """Check if x can be added to set S without violating the constraint."""
    S_list = list(S)
    m = len(S_list)
    # x ∤ a*b for all distinct a,b in S
    for i in range(m):
        a = S_list[i]
        for j in range(i + 1, m):
            b = S_list[j]
            if (a * b) % x == 0:
                return False
    # a ∤ x*b for all distinct a,b in S
    for i in range(m):
        a = S_list[i]
        for j in range(i + 1, m):
            b = S_list[j]
            if (x * b) % a == 0:
                return False
    return True

def _greedy_order(nums, order_func):
    """Greedy: add numbers in the order given by order_func."""
    S = set()
    for x in order_func(nums):
        if _can_add(S, x):
            S.add(x)
    return S

def _local_search(S, candidates, deadline):
    """Try to improve S by removing one element and adding others."""
    best = set(S)
    best_size = len(best)
    
    # Try all removals
    for rem in list(best):
        if time.time() > deadline:
            break
        S2 = set(best)
        S2.discard(rem)
        # Try to add candidates in various orders
        for x in candidates:
            if x in S2:
                continue
            if _can_add(S2, x):
                S2.add(x)
                # Try to add more
                for y in candidates:
                    if y in S2:
                        continue
                    if _can_add(S2, y):
                        S2.add(y)
                if len(S2) > best_size:
                    best = set(S2)
                    best_size = len(best)
    
    return best

def _iterated_local_search(nums, initial_set, deadline, max_iter=200):
    """Iterated Local Search with perturbation."""
    best_set = set(initial_set)
    best_size = len(best_set)
    candidates = [x for x in nums if x not in best_set]
    rng = random.Random(42)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Perturbation: randomly remove 1-3 elements
        current = set(best_set)
        n_remove = rng.randint(1, 3)
        to_remove = rng.sample(list(current), min(n_remove, len(current)))
        for x in to_remove:
            current.discard(x)
        
        # Add back candidates greedily
        cand_list = [x for x in nums if x not in current]
        rng.shuffle(cand_list)
        for x in cand_list:
            if _can_add(current, x):
                current.add(x)
        
        # Local search on current
        cand_for_ls = [x for x in nums if x not in current]
        current = _local_search(current, cand_for_ls, deadline)
        
        if len(current) > best_size:
            best_set = current
            best_size = len(best_size)
            candidates = [x for x in nums if x not in best_set]
    
    return best_set

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 55  # 55 second budget (leave 5s margin for 60s limit)
    
    n = P['n']
    nums = list(range(2, n + 1))
    
    # Get primes as initial solution
    primes = [x for x in nums if _is_prime(x)]
    best_set = set(primes)
    best_size = len(primes)
    
    # Try greedy with different orderings to get a better starting point
    orderings = [
        lambda nums: sorted(nums, key=lambda x: -x),  # descending
        lambda nums: sorted(nums, key=lambda x: x),    # ascending
        lambda nums: sorted(nums, key=lambda x: -sum(1 for d in range(2, int(x**0.5)+1) if x%d==0)),  # by number of divisors desc
        lambda nums: sorted(nums, key=lambda x: (x % 2, -x)),  # evens first then odds desc
    ]
    
    for order_func in orderings:
        if time.time() > _DEADLINE:
            break
        S_greedy = _greedy_order(nums, order_func)
        if len(S_greedy) > best_size:
            best_set = S_greedy
            best_size = len(S_greedy)
    
    # Use ILS from the best greedy solution
    if time.time() < _DEADLINE:
        best_set = _iterated_local_search(nums, best_set, _DEADLINE, max_iter=500)
        best_size = len(best_set)
    
    # Final local search polish
    if time.time() < _DEADLINE:
        candidates2 = [x for x in nums if x not in best_set]
        best_set = _local_search(best_set, candidates2, _DEADLINE)
        best_size = len(best_set)
    
    # Final verification
    S_list = sorted(best_set)
    if not _valid_set(S_list):
        # Fallback to primes
        S_list = sorted(primes)
    
    return list(S_list)
# EVOLVE-BLOCK-END
