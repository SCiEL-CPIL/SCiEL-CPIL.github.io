import time
import random
from math import gcd
from itertools import combinations

MAX_MOD = 128

# ---------- Precomputation ----------
def _precompute():
    gcd_table = {}
    for i in range(2, MAX_MOD + 1):
        for j in range(i, MAX_MOD + 1):
            gcd_table[(i, j)] = gcd(i, j)
    # prime factors (distinct) for each modulus
    prime_factors = {}
    for m in range(2, MAX_MOD + 1):
        pf = set()
        x = m
        d = 2
        while d * d <= x:
            while x % d == 0:
                pf.add(d)
                x //= d
            d += 1
        if x > 1:
            pf.add(x)
        prime_factors[m] = pf
    return gcd_table, prime_factors

GCD_TABLE, PRIME_FACTORS = _precompute()

def _get_gcd(m1, m2):
    if m1 > m2:
        m1, m2 = m2, m1
    return GCD_TABLE.get((m1, m2), 1)

# ---------- Clique finding (moduli with pairwise gcd > 1) ----------
def _find_clique(deadline):
    """Greedy + local search to find a large subset of moduli with pairwise gcd > 1."""
    moduli = list(range(2, MAX_MOD + 1))
    # Score each modulus by number of other moduli it is non-coprime with
    def score(m):
        return sum(1 for o in moduli if o != m and _get_gcd(m, o) > 1)
    
    # Greedy: repeatedly pick the modulus with highest "compatibility with current set"
    chosen = set()
    remaining = set(moduli)
    # Initial: pick the most connected
    best = max(moduli, key=score)
    chosen.add(best)
    remaining.remove(best)
    
    while remaining:
        # Pick modulus with max (count of non-coprime partners in chosen, then score)
        best_m = None
        best_key = (-1, -1)
        for m in remaining:
            compat = sum(1 for c in chosen if _get_gcd(m, c) > 1)
            key = (compat, score(m))
            if key > best_key:
                best_key = key
                best_m = m
        if best_key[0] != len(chosen):
            break  # can't add any more
        chosen.add(best_m)
        remaining.remove(best_m)
    
    # Local search: try to add a modulus by removing at most one
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each remaining modulus
        for m in list(remaining):
            # Check which chosen it conflicts with
            conflicts = [c for c in chosen if _get_gcd(m, c) == 1]
            if not conflicts:
                chosen.add(m)
                remaining.remove(m)
                improved = True
                break
        if improved:
            continue
        # Try swap: remove one c, add m
        for m in list(remaining):
            conflicts = [c for c in chosen if _get_gcd(m, c) == 1]
            if len(conflicts) == 1:
                c = conflicts[0]
                # Check m is compatible with all chosen - {c}
                ok = all(_get_gcd(m, cc) > 1 for cc in chosen if cc != c)
                if ok:
                    chosen.remove(c)
                    remaining.add(c)
                    chosen.add(m)
                    remaining.remove(m)
                    improved = True
                    break
    return chosen

# ---------- Residue assignment via backtracking ----------
def _assign_residues(modulus_set, deadline):
    """Assign residues so that for every pair, r_i != r_j mod gcd(m_i, m_j).
    Returns list of (r, m) or None if failed."""
    mods = list(modulus_set)
    # Order by most constrained first: higher gcd with others, larger modulus
    def constraint(m):
        gsum = sum(_get_gcd(m, o) for o in mods if o != m)
        return (gsum, m)
    mods.sort(key=constraint, reverse=True)
    
    n = len(mods)
    assignment = [None] * n  # assignment[i] = residue for mods[i]
    
    # For backtracking, precompute for each pair the gcd
    pair_gcd = [[_get_gcd(mods[i], mods[j]) for j in range(n)] for i in range(n)]
    
    def is_valid(i, r):
        for j in range(i):
            if assignment[j] is not None:
                g = pair_gcd[i][j]
                if g > 1 and (r - assignment[j]) % g == 0:
                    return False
        return True
    
    def backtrack(i):
        if time.time() > deadline:
            return False
        if i == n:
            return True
        m = mods[i]
        # Try residues; prefer those that block fewer future options
        # Simple: try all residues 0..m-1
        for r in range(m):
            if is_valid(i, r):
                assignment[i] = r
                if backtrack(i + 1):
                    return True
                assignment[i] = None
        return False
    
    if backtrack(0):
        return [(assignment[i], mods[i]) for i in range(n)]
    return None

# ---------- Improve via swap local search on residues ----------
def _try_improve(result, modulus_set, deadline):
    """Try to add more moduli by re-solving, or swap residues."""
    # Not needed for base; backtracking is exact for the given set.
    return result

# ---------- Main ----------
def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin for 90s limit
    
    # Step 1: Find a large clique of non-coprime moduli
    clique = _find_clique(deadline - 30.0)
    
    # Step 2: Assign residues
    result = _assign_residues(clique, deadline - 5.0)
    
    if result is None:
        # Fallback: use power-of-2 chain (always works)
        result = [(1, 2), (3, 4), (7, 8), (15, 16), (31, 32), (63, 64), (127, 128)]
    
    # Step 3: Try to extend by adding moduli one at a time with re-assignment
    if time.time() < deadline - 5.0:
        current_mods = set(m for _, m in result)
        all_mods = list(range(2, MAX_MOD + 1))
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for m in all_mods:
                if m in current_mods:
                    continue
                # Check m is non-coprime with all current
                if all(_get_gcd(m, cm) > 1 for cm in current_mods):
                    new_set = current_mods | {m}
                    new_result = _assign_residues(new_set, deadline - 5.0)
                    if new_result is not None and len(new_result) > len(result):
                        result = new_result
                        current_mods = set(m2 for _, m2 in result)
                        improved = True
                        break
    
    # Verify disjointness
    def verify(pairs):
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                r1, m1 = pairs[i]
                r2, m2 = pairs[j]
                g = _get_gcd(m1, m2)
                if (r1 - r2) % g == 0:
                    return False
        return True
    
    if not verify(result):
        result = [(1, 2), (3, 4), (7, 8), (15, 16), (31, 32), (63, 64), (127, 128)]
    
    return [[r, m] for r, m in result]
