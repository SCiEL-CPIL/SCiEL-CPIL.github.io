import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # All candidate moduli in [20, 200], sorted by 1/m descending (smallest m first)
    moduli = list(range(20, 201))
    moduli.sort(key=lambda m: -1.0/m)
    
    # Greedy: try to add each modulus with best residue
    # Key insight: for m1 | m2, constraint is r1 != r2 mod m1
    # For general m1, m2: constraint is r1 != r2 mod gcd(m1, m2)
    
    # We'll use a greedy approach with lookahead
    # For each modulus, try all residues and pick the one that
    # leaves the most options for future moduli
    
    def count_future_options(assign, m_new, r_new, future_moduli):
        """Count how many residues are available for each future modulus."""
        total = 0
        for m_f in future_moduli:
            if m_f in assign:
                continue
            # Count valid residues for m_f given current assignment + (m_new, r_new)
            valid_count = 0
            for r_f in range(m_f):
                ok = True
                # Check against existing assignment
                for m_e, r_e in assign.items():
                    g = math.gcd(m_f, m_e)
                    if r_f % g == r_e % g:
                        ok = False
                        break
                if not ok:
                    continue
                # Check against new placement
                g_new = math.gcd(m_f, m_new)
                if r_f % g_new == r_new % g_new:
                    ok = False
                if ok:
                    valid_count += 1
            total += valid_count
        return total
    
    def is_compatible(assign, m, r):
        """Check if (m, r) is compatible with all entries in assign."""
        for m2, r2 in assign.items():
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True
    
    # Greedy construction
    assign = {}  # m -> r
    remaining = set(moduli)
    
    # Process moduli in order of decreasing 1/m (smallest first)
    for m in moduli:
        if time.time() > deadline:
            break
        if m not in remaining:
            continue
        
        # Try to find a compatible residue
        best_r = None
        best_score = -1
        
        # For efficiency, if m is large, we can be smarter
        # But let's try a few promising residues first
        # Residues that are "independent" tend to work well
        
        # Try all residues but with early termination if we find a good one
        # Actually, for small m (20-50), try all. For larger, sample.
        
        candidates = list(range(m))
        # Sort candidates: prefer residues that don't conflict with small moduli
        # Actually, let's just try them all for small m
        if m <= 60:
            for r in candidates:
                if is_compatible(assign, m, r):
                    # Score: how many future options does this leave?
                    # Use a quick heuristic instead of full count for speed
                    # Count conflicts with already-placed moduli
                    conflict_count = 0
                    for m2, r2 in assign.items():
                        g = math.gcd(m, m2)
                        # This r is compatible, so no direct conflict
                        # But it constrains future moduli
                        pass
                    # Use flexibility: count how many residues would be blocked for gcd-related moduli
                    # Simplified: just pick the first compatible one, but prefer smaller residues
                    # Actually, let's use a better heuristic
                    best_r = r
                    break
        else:
            # For larger m, try a subset of residues
            # Residues 0, 1, 2, ... and also residues that are coprime to common factors
            tried = set()
            for r in list(range(min(m, 20))) + list(range(m-10, m)) if m > 20 else list(range(m)):
                if r in tried:
                    continue
                tried.add(r)
                if is_compatible(assign, m, r):
                    best_r = r
                    break
            if best_r is None:
                # Try more residues
                for r in range(m):
                    if r in tried:
                        continue
                    if is_compatible(assign, m, r):
                        best_r = r
                        break
        
        if best_r is not None:
            assign[m] = best_r
            remaining.discard(m)
        else:
            remaining.discard(m)  # Can't place this modulus
    
    # Build result
    result = [[m, r] for m, r in sorted(assign.items())]
    return result
