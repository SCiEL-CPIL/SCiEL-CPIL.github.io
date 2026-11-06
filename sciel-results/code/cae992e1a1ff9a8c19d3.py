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

def _can_add(r_new, m_new, existing):
    for (r, m) in existing:
        if not _disjoint(r_new, m_new, r, m):
            return False
    return True

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = _START_TIME + 85.0  # safety margin
    
    # The set of all even moduli from 2 to 128
    # This is a large set of pairwise non-coprime moduli
    candidate_moduli = [m for m in range(2, 129) if m % 2 == 0]
    
    # We will try to find residues for this set
    # If we fail, we will try smaller subsets
    
    # To help backtracking, we sort moduli.
    # Moduli with fewer divisors might be more flexible?
    # Or moduli with larger values?
    # Let's try sorting by descending value.
    # Or maybe by ascending number of divisors?
    
    # Let's use a randomized backtracking search
    # We'll try multiple random orderings if the first fails
    
    best_result = None
    
    # Precompute gcds for the candidate moduli
    n = len(candidate_moduli)
    gcd_table = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            gcd_table[i][j] = _gcd(candidate_moduli[i], candidate_moduli[j])
            gcd_table[j][i] = gcd_table[i][j]
            
    # Backtracking function
    # selected: list of (index, residue)
    # idx: current index in the permutation
    # perm: permutation of indices
    
    def backtrack(selected, perm, pos):
        if time.time() > deadline:
            return None
            
        if pos == len(perm):
            return selected[:]
            
        idx = perm[pos]
        m = candidate_moduli[idx]
        
        # Try all residues for this modulus
        # Shuffle residues to break symmetry / find solutions faster
        residues = list(range(m))
        random.shuffle(residues)
        
        for r in residues:
            # Check against all selected
            valid = True
            for (s_idx, s_r) in selected:
                g = gcd_table[idx][s_idx]
                if (r - s_r) % g == 0:
                    valid = False
                    break
            
            if valid:
                selected.append((idx, r))
                result = backtrack(selected, perm, pos + 1)
                if result is not None:
                    return result
                selected.pop()
                
        return None

    # Try to solve for the full set of evens
    # If it fails, we might need to drop some
    # However, given the structure, it's likely solvable.
    
    # To increase the chance of finding a solution quickly, we can use a heuristic ordering
    # Sort by descending modulus?
    # Let's try a few fixed orderings first, then random.
    
    orderings = [
        list(range(n)),
        list(range(n-1, -1, -1)),
        sorted(range(n), key=lambda i: candidate_moduli[i]),
        sorted(range(n), key=lambda i: -candidate_moduli[i]),
    ]
    
    # Add a few random shuffles
    for _ in range(20):
        p = list(range(n))
        random.shuffle(p)
        orderings.append(p)
        
    for perm in orderings:
        if time.time() > deadline:
            break
            
        result = backtrack([], perm, 0)
        if result is not None:
            # Convert to [residue, modulus] pairs
            best_result = [[r, candidate_moduli[i]] for (i, r) in result]
            # We found a solution for all 63 evens. 
            # Can we do better? 
            # The max size of a non-coprime subset is likely 63.
            # So this is optimal.
            break
            
    # If we didn't find a solution for all 63, we should try to find the largest subset
    # But for now, if we have a result, return it.
    # If not, we need a fallback.
    
    if best_result is None:
        # Fallback: try smaller sets or a simpler construction
        # The power of 2 construction gives 7 classes.
        # We can try to greedily add moduli.
        
        selected = []
        # Try all moduli in descending order of "complexity" or just random
        moduli_to_try = list(range(2, 129))
        random.shuffle(moduli_to_try)
        
        for m in moduli_to_try:
            if time.time() > deadline:
                break
            # Check if m is coprime to any selected modulus?
            # If yes, skip.
            skip = False
            for (r, sm) in selected:
                if _gcd(m, sm) == 1:
                    skip = True
                    break
            if skip:
                continue
                
            # Try to find a residue
            for r in range(m):
                if _can_add(r, m, selected):
                    selected.append((r, m))
                    break
                    
        if selected:
            best_result = [[r, m] for (r, m) in selected]
        else:
            # Absolute fallback
            best_result = [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
            
    return best_result
