import time
import math
import random
from math import gcd

MAXIMUM = 128
DEADLINE = 90.0
SAFETY_MARGIN = 2.0

def _check_pair(r1, m1, r2, m2):
    """Two residue classes are disjoint iff r1 mod gcd(m1,m2) != r2 mod gcd(m1,m2)."""
    g = gcd(m1, m2)
    return r1 % g != r2 % g

def _is_valid_solution(solution):
    if not solution:
        return False
    seen_moduli = set()
    for r, m in solution:
        if m < 2 or m > MAXIMUM:
            return False
        if m in seen_moduli:
            return False
        seen_moduli.add(m)
        if r < 0 or r >= m:
            return False
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            if not _check_pair(solution[i][0], solution[i][1], solution[j][0], solution[j][1]):
                return False
    return True

def _greedy_construct(mod_order, deadline):
    solution = []
    for m in mod_order:
        if time.time() > deadline:
            break
        for r in range(m):
            compatible = True
            for (r_prev, m_prev) in solution:
                g = gcd(m, m_prev)
                if r % g == r_prev % g:
                    compatible = False
                    break
            if compatible:
                solution.append([r, m])
                break
    return solution

def _try_add_modulus(solution, m, deadline):
    for r in range(m):
        if time.time() > deadline:
            return None
        candidate = solution + [[r, m]]
        if _is_valid_solution(candidate):
            return candidate
    return None

def _local_search(solution, deadline):
    if not solution:
        return solution
    
    best = [list(x) for x in solution]
    all_moduli = list(range(2, MAXIMUM + 1))
    
    # Phase 1: greedily add any unused modulus
    used = set(m for _, m in best)
    for m in all_moduli:
        if time.time() > deadline:
            break
        if m in used:
            continue
        result = _try_add_modulus(best, m, deadline)
        if result is not None:
            best = result
            used.add(m)
    
    # Phase 2: try removing one and adding two different ones
    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(best)):
            if time.time() > deadline:
                break
            without_i = [list(x) for x in best[:i]] + [list(x) for x in best[i+1:]]
            used_m = set(x[1] for x in without_i)
            m_i = best[i][1]
            
            for r_i in range(m_i):
                if time.time() > deadline:
                    break
                candidate = without_i + [[r_i, m_i]]
                if not _is_valid_solution(candidate):
                    continue
                for m_add in all_moduli:
                    if time.time() > deadline:
                        break
                    if m_add in used_m or m_add == m_i:
                        continue
                    result = _try_add_modulus(candidate, m_add, deadline)
                    if result is not None:
                        if len(result) > len(best):
                            best = result
                            improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    all_moduli = list(range(2, MAXIMUM + 1))
    best_solution = []
    
    # Multiple greedy runs with different orderings
    orderings = [
        all_moduli,
        list(reversed(all_moduli)),
        sorted(all_moduli, key=lambda m: m.bit_count(), reverse=True),
        sorted(all_moduli, key=lambda m: -m.bit_count()),
    ]
    
    for mod_order in orderings:
        if time.time() > deadline:
            break
        sol = _greedy_construct(mod_order, deadline)
        if len(sol) > len(best_solution):
            best_solution = sol
    
    # Random orderings
    for _ in range(50):
        if time.time() > deadline:
            break
        random_order = all_moduli[:]
        random.shuffle(random_order)
        sol = _greedy_construct(random_order, deadline)
        if len(sol) > len(best_solution):
            best_solution = sol
    
    # Local search
    if best_solution:
        improved = _local_search(best_solution, deadline)
        if len(improved) > len(best_solution):
            best_solution = improved
    
    if not best_solution:
        return [[0, 2]]
    
    return best_solution
