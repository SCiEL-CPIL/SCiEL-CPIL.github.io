import time
import math
from math import gcd

MAXIMUM = 128
DEADLINE = 90.0
SAFETY_MARGIN = 1.0

def _is_valid_solution(solution):
    """Check if a solution is valid: distinct moduli in [2,128], canonical residues, pairwise disjoint."""
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
            r1, m1 = solution[i]
            r2, m2 = solution[j]
            g = gcd(m1, m2)
            if r1 % g == r2 % g:
                return False
    return True

def _backtrack_search(candidates, current_solution, deadline, best_solution):
    """
    Recursively search for a larger solution.
    """
    if time.time() > deadline:
        return best_solution
    
    # Pruning: even if we add all remaining, we can't beat best
    if len(current_solution) + len(candidates) <= len(best_solution):
        return best_solution
        
    if len(current_solution) > len(best_solution):
        best_solution = [list(x) for x in current_solution]
    
    if not candidates:
        return best_solution
    
    m = candidates[0]
    rest = candidates[1:]
    
    # Check if m can be added at all (gcd > 1 with all existing)
    compatible_gcds = True
    g_list = []
    for r_prev, m_prev in current_solution:
        g = gcd(m, m_prev)
        if g == 1:
            compatible_gcds = False
            break
        g_list.append(g)
        
    if compatible_gcds:
        # Try to find a residue
        found = False
        # We can try a few residues, not just the first one, to allow backtracking to work better?
        # But trying all residues is expensive.
        # Let's try the first valid one.
        for r in range(m):
            ok = True
            for idx, g in enumerate(g_list):
                r_prev = current_solution[idx][0]
                if r % g == r_prev % g:
                    ok = False
                    break
            if ok:
                current_solution.append([r, m])
                best_solution = _backtrack_search(rest, current_solution, deadline, best_solution)
                current_solution.pop()
                found = True
                break # Only try the first valid residue for this modulus in this branch
        
    # Try skipping m
    best_solution = _backtrack_search(rest, current_solution, deadline, best_solution)
    
    return best_solution

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_solution = []
    
    # Strategy 1: The seed's power-of-2 approach
    seed_solution = [[(1 << k) - 1, 1 << (k + 1)] for k in range(6)]
    if _is_valid_solution(seed_solution):
        best_solution = [list(x) for x in seed_solution]
        
    # Strategy 2: Backtracking with different orderings
    all_moduli = list(range(2, MAXIMUM + 1))
    
    # Order 1: Descending value
    order1 = sorted(all_moduli, reverse=True)
    # Order 2: Ascending value
    order2 = sorted(all_moduli)
    # Order 3: By number of divisors (descending)
    def num_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    order3 = sorted(all_moduli, key=num_divisors, reverse=True)
    # Order 4: Random
    import random
    random.seed(42)
    order4 = all_moduli.copy()
    random.shuffle(order4)
    
    for order in [order3, order1, order2, order4]:
        if time.time() > deadline:
            break
        current = []
        best_solution = _backtrack_search(order, current, deadline, best_solution)
        
    return best_solution if best_solution else [[0, 2]]
