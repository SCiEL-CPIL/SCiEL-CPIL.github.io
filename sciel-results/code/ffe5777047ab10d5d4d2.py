import math
import time
import random

MAX_MOD = 128
DEADLINE = 60.0
MARGIN = 5.0

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - MARGIN
    
    all_mods = list(range(2, MAX_MOD + 1))
    
    # Pre-compute gcds for all pairs
    gcd_table = {}
    for m1 in all_mods:
        for m2 in all_mods:
            if m1 != m2:
                gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    def fast_disjoint(r1, m1, r2, m2):
        g = gcd_table[(m1, m2)]
        return (r2 - r1) % g != 0

    def _solve_fast(moduli, t_limit, seed_val):
        random.seed(seed_val)
        n = len(moduli)
        if n == 0:
            return []
        
        residues = [None] * n
        
        def backtrack(idx):
            if time.time() > t_limit:
                return False
            if idx == n:
                return True
            
            m = moduli[idx]
            residues_to_try = list(range(m))
            random.shuffle(residues_to_try)
            
            for r in residues_to_try:
                valid = True
                for j in range(idx):
                    if not fast_disjoint(r, m, residues[j], moduli[j]):
                        valid = False
                        break
                if valid:
                    residues[idx] = r
                    if backtrack(idx + 1):
                        return True
                    residues[idx] = None
            return False
        
        if backtrack(0):
            return [(residues[i], moduli[i]) for i in range(n)]
        return None

    # Strategy:
    # 1. Identify moduli that are "2 * odd" (e.g., 2, 6, 10, 14...).
    #    If we have three such moduli m1, m2, m3, they all share gcd 2 with each other.
    #    Let's check if we can have 3.
    #    C1: r1 mod m1, C2: r2 mod m2, C3: r3 mod m3.
    #    Disjointness: (r2-r1)%2 != 0, (r3-r1)%2 != 0, (r3-r2)%2 != 0.
    #    This implies r1, r2, r3 must have distinct parities. Impossible since there are only 2 parities.
    #    So we can have at most 2 moduli of the form 2*odd.
    
    e_odd = [m for m in all_mods if m % 2 == 0 and (m // 2) % 2 == 1]
    # e_odd = [2, 6, 10, 14, ..., 126]
    # Size: 32.
    
    # We must drop at least 30 of these.
    # Which ones to keep?
    # Keeping smaller ones might be better? Or larger?
    # Let's try keeping the 2 smallest: 2 and 6.
    # Or maybe 2 and 126?
    # Let's just pick 2 arbitrarily, say the first two: 2, 6.
    # But maybe we can keep different pairs and solve the rest.
    
    # The set of all moduli except 30 of e_odd has size 127 - 30 = 97.
    # Is 97 solvable?
    # The remaining moduli include:
    # - All odd numbers (63 of them).
    # - All evens that are multiples of 4 (32 of them: 4, 8, ..., 128).
    # - The 2 kept e_odd (e.g., 2, 6).
    
    # Let's try to solve this specific subset.
    # We can try a few different pairs of kept e_odd to see if one works faster or if they all work.
    
    best_pairs = []
    best_len = 0
    
    # Candidate sets to try
    # 1. Keep {2, 6}
    # 2. Keep {2, 126}
    # 3. Keep {6, 10}
    # etc.
    
    # Let's generate a few candidate subsets
    candidates = []
    
    # Base set: all moduli
    # Remove 30 from e_odd
    # We can iterate over pairs in e_odd to keep
    
    # To save time, let's just try a few specific pairs.
    # Pair (2, 6)
    keep1 = {2, 6}
    mods1 = [m for m in all_mods if m not in e_odd or m in keep1]
    candidates.append(mods1)
    
    # Pair (2, 126)
    keep2 = {2, 126}
    mods2 = [m for m in all_mods if m not in e_odd or m in keep2]
    candidates.append(mods2)
    
    # Pair (6, 10)
    keep3 = {6, 10}
    mods3 = [m for m in all_mods if m not in e_odd or m in keep3]
    candidates.append(mods3)
    
    # Try solving each candidate
    for mods in candidates:
        if time.time() > deadline - 5.0:
            break
            
        # Try a few random seeds/orderings for this specific subset
        for seed in range(10):
            if time.time() > deadline - 1.0:
                break
                
            random.seed(seed)
            shuffled_mods = mods[:]
            random.shuffle(shuffled_mods)
            
            pairs = _solve_fast(shuffled_mods, deadline - 1.0, seed)
            
            if pairs is not None:
                if len(pairs) > best_len:
                    best_pairs = pairs
                    best_len = len(pairs)
                break # Found a solution for this subset, move to next candidate or finish?
            else:
                # If it fails, maybe try again with different seed?
                # The loop handles that.
                pass
        
        # If we found a solution, is it maximal?
        # We know we can't have more than 2 from e_odd.
        # So the max size is 97.
        # If we found 97, we are done.
        if best_len == 97:
            break
    
    # If we didn't find a solution of size 97, maybe the subset is not solvable?
    # Or the solver is too slow?
    # Let's fall back to greedy on the full set, which might find a large but not maximal set.
    # But greedy on full set will likely get stuck on e_odd conflicts early.
    # A better fallback: Greedy on the "safe" subset (all non-e-odd + 2 e-odd).
    
    if best_len < 97:
        # Try greedy on mods1
        for seed in range(10):
            if time.time() > deadline - 0.5:
                break
            random.seed(seed)
            mods = mods1[:]
            random.shuffle(mods)
            
            pairs = []
            for m in mods:
                if time.time() > deadline - 0.2:
                    break
                found = False
                r_candidates = list(range(m))
                random.shuffle(r_candidates)
                for r in r_candidates:
                    ok = True
                    for (r2, m2) in pairs:
                        if not fast_disjoint(r, m, r2, m2):
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
        # Fallback
        best_pairs = [(0, 2)]
        
    return [[r, m] for r, m in best_pairs]
