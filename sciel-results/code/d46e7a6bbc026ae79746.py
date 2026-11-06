import math
import time
import random

_START_TIME = None

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _disjoint(r1, m1, r2, m2):
    """Check if residue classes (r1 mod m1) and (r2 mod m2) are disjoint."""
    g = _gcd(m1, m2)
    if g == 1:
        return False  # always intersect
    return (r2 - r1) % g != 0

def _can_add(r_new, m_new, selected, gcd_table):
    """Check if new (r_new, m_new) is disjoint from all selected classes."""
    for (r, m) in selected:
        g = gcd_table.get((m_new, m), 1)
        if g == 1:
            return False
        if (r_new - r) % g == 0:
            return False
    return True

def _solve_family(moduli, deadline):
    """
    Attempt to find a valid residue assignment for the given list of moduli.
    Uses a greedy randomized restart strategy with backtracking if needed.
    """
    if not moduli:
        return []
    
    # Sort moduli by size (smaller moduli have fewer residues, so they are more constrained)
    sorted_moduli = sorted(moduli)
    n = len(sorted_moduli)
    
    # Precompute gcds for this specific set
    gcds = [[1]*n for _ in range(n)]
    for i in range(n):
        for j in range(i):
            g = _gcd(sorted_moduli[i], sorted_moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
    
    # Backtracking solver
    def backtrack(idx, current_residues):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        m = sorted_moduli[idx]
        # Try residues in random order
        residues = list(range(m))
        random.shuffle(residues)
        
        for r in residues:
            valid = True
            for j in range(idx):
                g = gcds[idx][j]
                if g == 1:
                    valid = False
                    break
                if (r - current_residues[j]) % g == 0:
                    valid = False
                    break
            
            if valid:
                current_residues[idx] = r
                if backtrack(idx + 1, current_residues):
                    return True
        
        return False

    # Try multiple random seeds/orderings
    best_solution = None
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        current_residues = [0] * n
        if backtrack(0, current_residues):
            # Found a solution
            solution = []
            for i in range(n):
                solution.append((current_residues[i], sorted_moduli[i]))
            if best_solution is None or len(solution) > len(best_solution):
                best_solution = solution
            break # If we found a full solution, it's optimal for this family
    
    return best_solution if best_solution else []

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = _START_TIME + 85.0  # safety margin
    
    all_moduli = list(range(2, 129))
    
    best_result = []
    
    # Family 1: All even numbers
    even_moduli = [m for m in all_moduli if m % 2 == 0]
    res_even = _solve_family(even_moduli, deadline)
    if len(res_even) > len(best_result):
        best_result = res_even
        
    # Family 2: All multiples of 3
    mult3_moduli = [m for m in all_moduli if m % 3 == 0]
    res_mult3 = _solve_family(mult3_moduli, deadline)
    if len(res_mult3) > len(best_result):
        best_result = res_mult3
        
    # Family 3: All multiples of 5
    mult5_moduli = [m for m in all_moduli if m % 5 == 0]
    res_mult5 = _solve_family(mult5_moduli, deadline)
    if len(res_mult5) > len(best_result):
        best_result = res_mult5
        
    # Fallback: General greedy search if specific families failed
    if not best_result:
        gcd_table = {}
        for m1 in all_moduli:
            for m2 in all_moduli:
                if m1 != m2:
                    gcd_table[(m1, m2)] = _gcd(m1, m2)
        
        best_count = 0
        for attempt in range(100):
            if time.time() > deadline:
                break
            random.seed(attempt)
            perm = all_moduli[:]
            random.shuffle(perm)
            selected = []
            for m in perm:
                if time.time() > deadline:
                    break
                skip = False
                for (r, sm) in selected:
                    if _gcd(m, sm) == 1:
                        skip = True
                        break
                if skip:
                    continue
                residues = list(range(m))
                random.shuffle(residues)
                for r in residues:
                    if _can_add(r, m, selected, gcd_table):
                        selected.append((r, m))
                        break
            if len(selected) > best_count:
                best_count = len(selected)
                best_result = selected[:]
                
    # Final fallback
    if not best_result:
        best_result = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
        
    return [[r, m] for r, m in best_result]
