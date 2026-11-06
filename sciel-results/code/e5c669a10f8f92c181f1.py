import math
import random
import time

MAX_MOD = 128
DEADLINE_MARGIN = 5.0

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _solve_csp(moduli, time_limit):
    """
    Solves the CSP for the given moduli.
    Returns list of (r, m) or None.
    """
    n = len(moduli)
    if n == 0:
        return []
    
    # Order moduli to help backtracking.
    # Moduli with more constraints (higher degree in gcd graph) first?
    # Or just sort by value?
    # Let's try sorting by decreasing "density" of constraints.
    # For simplicity, sort by value.
    indices = list(range(n))
    
    # Precompute gcds
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            gcds[i][j] = math.gcd(moduli[i], moduli[j])
            gcds[j][i] = gcds[i][j]
            
    residues = [None] * n
    
    def backtrack(idx):
        if time.time() > time_limit:
            return False
            
        if idx == n:
            return True
            
        m = moduli[idx]
        # Try residues in random order
        residues_to_try = list(range(m))
        random.shuffle(residues_to_try)
        
        for r in residues_to_try:
            valid = True
            for j in range(idx):
                if not _disjoint(r, m, residues[j], moduli[j]):
                    valid = False
                    break
            if valid:
                residues[idx] = r
                if backtrack(idx + 1):
                    return True
        residues[idx] = None
        return False

    # Try a few restarts
    for _ in range(10):
        random.shuffle(indices)
        # Reorder moduli? 
        # Actually, the order in `moduli` list is fixed for the check.
        # We should shuffle the `moduli` list itself or the indices used to access it.
        # For simplicity, just shuffle the moduli list passed to the function?
        # No, the function takes a fixed list.
        # Let's shuffle the moduli list in the caller.
        
    if backtrack(0):
        return [(residues[i], moduli[i]) for i in range(n)]
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 60.0 - DEADLINE_MARGIN
    
    all_mods = list(range(2, MAX_MOD + 1))
    
    # Identify E_odd: evens that are 2 * odd
    # These are the ones causing the parity conflict if 3 or more are present.
    e_odd = [m for m in all_mods if m % 2 == 0 and (m // 2) % 2 == 1]
    
    # We can keep at most 2 from e_odd.
    # Keep the smallest 2 to minimize modulus size (easier for solver? maybe).
    e_odd_to_keep = sorted(e_odd)[:2]
    
    # Construct candidate set
    candidate_mods = [m for m in all_mods if m not in e_odd or m in e_odd_to_keep]
    
    # Shuffle to help backtracking
    random.shuffle(candidate_mods)
    
    best_pairs = []
    best_len = 0
    
    # Try to solve for the full candidate set
    # If it fails, we might need to remove more moduli.
    # But based on theory, this set should be solvable.
    
    # To be safe, if it fails, we can try removing one by one.
    # But let's try the full set first with a time limit.
    
    # We will run the solver.
    # Note: _solve_csp as written above doesn't use the shuffled order correctly if we don't pass the shuffled list.
    # Let's pass the shuffled list.
    
    # We'll do a few attempts with different shuffles
    for attempt in range(5):
        if time.time() > deadline - 5.0:
            break
        random.shuffle(candidate_mods)
        pairs = _solve_csp(candidate_mods, deadline - 5.0)
        if pairs is not None:
            best_pairs = pairs
            best_len = len(pairs)
            break
            
    # If we failed to find a solution for the full set (unlikely), 
    # we can try a greedy approach on the full set to get a partial solution.
    if best_len < len(candidate_mods):
        # Greedy on all_mods
        greedy_mods = all_mods[:]
        random.shuffle(greedy_mods)
        pairs = []
        for m in greedy_mods:
            if time.time() > deadline - 1.0:
                break
            found = False
            r_candidates = list(range(m))
            random.shuffle(r_candidates)
            for r in r_candidates:
                ok = True
                for (r2, m2) in pairs:
                    if not _disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    pairs.append((r, m))
                    found = True
                    break
            if not found:
                continue
        if len(pairs) > best_len:
            best_pairs = pairs
            best_len = len(pairs)
            
    if not best_pairs:
        best_pairs = [(0, 2)]
        
    return [[r, m] for r, m in best_pairs]
