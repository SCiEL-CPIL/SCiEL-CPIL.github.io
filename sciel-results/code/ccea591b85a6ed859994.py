"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(n + 1) if sieve[i]]

def _compute_L(classes, cap=100000):
    """Compute the largest L such that every integer 1..L is covered by some class."""
    # classes: list of [p, r]
    # An integer x is covered if there exists (p, r) such that x % p == r.
    # We need to find the first gap.
    
    if not classes:
        return 0
    
    # Use a bytearray for coverage up to cap
    # Actually, we can be smarter: find the first uncovered integer.
    # Let's just iterate from 1 upward and check each integer.
    # But that could be slow for large L.
    
    # Alternative: use a sieve-like approach.
    # Mark all covered integers up to cap.
    covered = bytearray(cap + 1)
    for p, r in classes:
        # Mark r, r+p, r+2p, ... up to cap
        # But we only care about 1..cap
        start = r
        if start == 0:
            start = p  # since we want 1..cap, and 0 is not in range
        # Actually, x % p == r. If r == 0, then x is a multiple of p.
        # The smallest positive x with x % p == 0 is p.
        # If r > 0, the smallest positive x is r (if r >= 1).
        
        # Let's just mark all x in [1, cap] such that x % p == r.
        # x = r + k*p for k >= 0, and x >= 1.
        if r == 0:
            first = p
        else:
            first = r
        
        if first > cap:
            continue
            
        for x in range(first, cap + 1, p):
            covered[x] = 1
    
    # Now find the first 0 in covered[1:]
    for i in range(1, cap + 1):
        if not covered[i]:
            return i - 1
    return cap

def _try_improve(classes, primes_available, budget, L_current, deadline):
    """Try to improve the current solution by swapping or adding classes."""
    # This is a simplified version; in practice, we'd do more sophisticated search.
    # For now, let's try adding a new class if budget allows, or swapping.
    
    best_L = L_current
    best_classes = [list(c) for c in classes]
    
    # Try adding each available prime with the best residue
    for p in primes_available:
        if time.time() > deadline:
            break
        # Check if we can add this prime
        # We need sum(1/p) + 1/p_new <= 1
        # budget is the remaining reciprocal budget
        if budget >= Fraction(1, p):
            # Try all residues for this prime
            for r in range(p):
                new_classes = best_classes + [[p, r]]
                new_L = _compute_L(new_classes, 100000)
                if new_L > best_L:
                    best_L = new_L
                    best_classes = new_classes
    
    return best_classes, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140 seconds to be safe
    
    primes = _primes_upto(1000)
    
    # Start with the seed solution
    seed_primes = [p for p in primes if p >= 13]
    classes = [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]
    
    # Compute initial budget
    total = sum(Fraction(1, p) for p in seed_primes)
    budget = Fraction(1) - total
    
    L = _compute_L(classes, 100000)
    
    # Now try to improve
    # Available primes are those not in the current set
    used_primes = set(p for p, r in classes)
    available_primes = [p for p in primes if p not in used_primes]
    
    # Sort available primes by size (smaller primes give more coverage per budget)
    available_primes.sort()
    
    # Try to add primes
    for _ in range(20):  # Limit iterations
        if time.time() > deadline:
            break
        
        improved = False
        for p in available_primes:
            if time.time() > deadline:
                break
            if budget >= Fraction(1, p):
                # Try to find the best residue for this prime
                best_r = None
                best_L_for_p = L
                for r in range(p):
                    new_classes = classes + [[p, r]]
                    new_L = _compute_L(new_classes, 100000)
                    if new_L > best_L_for_p:
                        best_L_for_p = new_L
                        best_r = r
                
                if best_r is not None and best_L_for_p > L:
                    classes.append([p, best_r])
                    budget -= Fraction(1, p)
                    L = best_L_for_p
                    available_primes.remove(p)
                    improved = True
                    break
        
        if not improved:
            break
    
    return classes
# EVOLVE-BLOCK-END
