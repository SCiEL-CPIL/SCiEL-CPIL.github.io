import math
import time

MAX_MOD = 128

def gcd(a, b):
    return math.gcd(a, b)

def is_disjoint(r1, m1, r2, m2):
    g = gcd(m1, m2)
    if g == 1:
        return False
    return ((r2 - r1) % g) != 0

def verify_solution(pairs):
    n = len(pairs)
    for i in range(n):
        r1, m1 = pairs[i]
        if m1 < 2 or m1 > MAX_MOD or r1 < 0 or r1 >= m1:
            return False
        for j in range(i + 1, n):
            r2, m2 = pairs[j]
            if m1 == m2:
                return False
            if not is_disjoint(r1, m1, r2, m2):
                return False
    return True

def greedy_construct(moduli, deadline):
    """
    Greedy: assign residues one at a time.
    For each modulus, pick a residue that is disjoint from all previously assigned.
    Order moduli by decreasing "constraint power" (larger moduli first, or those with many shared factors).
    """
    if time.time() > deadline:
        return []
    
    # Sort moduli: process those with larger values first (more residue options)
    # Actually, process in an order that makes constraints tighter later
    # Let's try: sort by modulus descending
    sorted_mods = sorted(moduli, reverse=True)
    
    pairs = []
    for m in sorted_mods:
        if time.time() > deadline:
            break
        
        # Find a residue r in [0, m-1] that is disjoint from all existing pairs
        found = False
        for r in range(m):
            ok = True
            for (r2, m2) in pairs:
                if not is_disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                pairs.append([r, m])
                found = True
                break
        
        if not found:
            # Try different residue orderings
            for r in range(m - 1, -1, -1):
                ok = True
                for (r2, m2) in pairs:
                    if not is_disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    pairs.append([r, m])
                    found = True
                    break
    
    return pairs

def backtracking_construct(moduli, deadline):
    """
    Backtracking search to maximize the number of classes.
    Tries to include as many moduli as possible.
    """
    if time.time() > deadline:
        return []
    
    n = len(moduli)
    # Precompute gcd matrix
    gcd_mat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            gcd_mat[i][j] = gcd(moduli[i], moduli[j])
            gcd_mat[j][i] = gcd_mat[i][j]
    
    best_solution = []
    
    def backtrack(idx, assigned, residues):
        nonlocal best_solution
        if time.time() > deadline:
            return
        
        if len(assigned) > len(best_solution):
            best_solution = list(assigned)
        
        if idx == n:
            return
        
        # Option 1: skip moduli[idx]
        backtrack(idx + 1, assigned, residues)
        
        # Option 2: try to assign a residue to moduli[idx]
        m = moduli[idx]
        for r in range(m):
            if time.time() > deadline:
                return
            ok = True
            for (r2, m2) in assigned:
                if not is_disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                assigned.append([r, m])
                backtrack(idx + 1, assigned, residues)
                assigned.pop()
    
    backtrack(0, [], {})
    return best_solution

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # Safety margin before 90s kill
    
    # Strategy: Use all even moduli (64 moduli from 2 to 128)
    # These have large pairwise gcds, making the disjointness condition more restrictive
    # but also meaning each residue choice eliminates fewer options for others
    
    even_moduli = list(range(2, MAX_MOD + 1, 2))
    
    # Try greedy first
    sol = greedy_construct(even_moduli, deadline)
    if sol and verify_solution(sol):
        return sol
    
    # Try multiples of 3
    mult3_moduli = list(range(3, MAX_MOD + 1, 3))
    sol = greedy_construct(mult3_moduli, deadline)
    if sol and verify_solution(sol):
        return sol
    
    # Try a mix: all moduli that are multiples of 2 or 3
    mixed_moduli = list(range(2, MAX_MOD + 1))
    mixed_moduli = [m for m in mixed_moduli if m % 2 == 0 or m % 3 == 0]
    sol = greedy_construct(mixed_moduli, deadline)
    if sol and verify_solution(sol):
        return sol
    
    # Fallback: powers of 2 with r = m/2
    fallback = [[m//2, m] for m in [2, 4, 8, 16, 32, 64, 128]]
    return fallback
