"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

# Precompute prime factors for all numbers 1..511
def _compute_prime_factors(n_max):
    pf = [[] for _ in range(n_max + 1)]
    for p in range(2, n_max + 1):
        if not pf[p]:
            for m in range(p, n_max + 1, p):
                pf[m].append(p)
    return pf

def _is_pairwise_coprime(selected, pf):
    """Check if selected list is pairwise coprime using prime factors."""
    seen_primes = set()
    for a in selected:
        for p in pf[a]:
            if p in seen_primes:
                return False
            seen_primes.add(p)
    return True

def _score(selected):
    """Compute exact score sum 1/(512-a) as Fraction."""
    total = Fraction(0, 1)
    for a in selected:
        total += Fraction(1, 512 - a)
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    pf = _compute_prime_factors(511)
    
    # Greedy initialization: pick numbers that maximize gain per "cost"
    # We want to maximize sum 1/(512-a), so we prefer large a (small denominator)
    # but must maintain pairwise coprimality
    
    # Strategy: greedy by descending value of 1/(512-a), i.e., descending a
    # but use smarter selection considering conflicts
    
    selected = []
    selected_set = set()
    
    # First pass: greedy from high a to low a
    for a in range(511, 0, -1):
        if time.time() > deadline:
            break
        # Check if a is coprime to all selected
        compatible = True
        for p in pf[a]:
            # Check if any selected number shares prime p
            for b in selected:
                if p in pf[b]:
                    compatible = False
                    break
                if not compatible:
                    break
        if compatible:
            selected.append(a)
            selected_set.add(a)
    
    # Now do local search: try 1-for-1 swaps
    best_score = _score(selected)
    best_selected = list(selected)
    
    # Create a faster conflict check using a prime->count map
    def build_prime_map(sel):
        pm = {}
        for a in sel:
            for p in pf[a]:
                pm[p] = pm.get(p, 0) + 1
        return pm
    
    # Local search with 1-for-1 and 1-for-2 moves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        pm = build_prime_map(best_selected)
        
        # Try 1-for-1: remove one, add another
        for i in range(len(best_selected)):
            if time.time() > deadline:
                break
            a_remove = best_selected[i]
            # Temporarily remove
            for p in pf[a_remove]:
                pm[p] -= 1
                if pm[p] == 0:
                    del pm[p]
            
            # Try adding candidates not in best_selected
            best_gain = None
            best_add = None
            for a_add in range(511, 0, -1):
                if a_add in best_selected:
                    continue
                # Check if a_add can be added
                can_add = True
                for p in pf[a_add]:
                    if p in pm:
                        can_add = False
                        break
                if can_add:
                    gain = Fraction(1, 512 - a_add) - Fraction(1, 512 - a_remove)
                    if best_gain is None or gain > best_gain:
                        best_gain = gain
                        best_add = a_add
            
            # Restore
            for p in pf[a_remove]:
                pm[p] = pm.get(p, 0) + 1
            
            if best_add is not None and best_gain > 0:
                # Apply the swap
                best_selected.remove(a_remove)
                best_selected.append(best_add)
                best_score += best_gain
                improved = True
                break  # Restart loop after improvement
        
        # Try 1-for-2: remove one, add two
        if time.time() > deadline:
            break
        for i in range(len(best_selected)):
            if time.time() > deadline:
                break
            a_remove = best_selected[i]
            # Temporarily remove
            for p in pf[a_remove]:
                pm[p] -= 1
                if pm[p] == 0:
                    del pm[p]
            
            # Try pairs of candidates
            candidates = [a for a in range(511, 0, -1) if a not in best_selected]
            best_gain = None
            best_pair = None
            n_cand = len(candidates)
            for idx1 in range(n_cand):
                if time.time() > deadline:
                    break
                a1 = candidates[idx1]
                # Check a1
                can_add1 = True
                for p in pf[a1]:
                    if p in pm:
                        can_add1 = False
                        break
                if not can_add1:
                    continue
                # Temporarily add a1's primes
                for p in pf[a1]:
                    pm[p] = pm.get(p, 0) + 1
                
                for idx2 in range(idx1 + 1, n_cand):
                    a2 = candidates[idx2]
                    # Check a2 against updated pm
                    can_add2 = True
                    for p in pf[a2]:
                        if p in pm:
                            can_add2 = False
                            break
                    if can_add2:
                        gain = Fraction(1, 512 - a1) + Fraction(1, 512 - a2) - Fraction(1, 512 - a_remove)
                        if best_gain is None or gain > best_gain:
                            best_gain = gain
                            best_pair = (a1, a2)
                
                # Restore a1's primes
                for p in pf[a1]:
                    pm[p] -= 1
                    if pm[p] == 0:
                        del pm[p]
            
            # Restore removed
            for p in pf[a_remove]:
                pm[p] = pm.get(p, 0) + 1
            
            if best_pair is not None and best_gain > 0:
                best_selected.remove(a_remove)
                best_selected.extend(best_pair)
                best_score += best_gain
                improved = True
                break  # Restart after improvement
    
    return best_selected
# EVOLVE-BLOCK-END
