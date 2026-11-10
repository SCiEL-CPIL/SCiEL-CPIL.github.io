"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
import math

def _sieve_primes(limit):
    """Generate all primes up to limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def _build_coverage(classes, cap=100000):
    """Build a coverage count array for 1..cap."""
    cov = [0] * (cap + 1)
    for p, r in classes:
        r_mod = r % p
        # Mark all n in [1, cap] with n % p == r_mod
        # First n >= 1 with n % p == r_mod:
        # n = r_mod if r_mod >= 1, else p (if r_mod == 0, n = p, 2p, ...)
        if r_mod == 0:
            start = p
        else:
            start = r_mod
        for n in range(start, cap + 1, p):
            cov[n] += 1
    return cov

def _compute_L_from_cov(cov, cap=100000):
    """Find largest L such that all 1..L have cov[n] > 0."""
    for L in range(1, cap + 1):
        if cov[L] == 0:
            return L - 1
    return cap

def _add_class(cov, p, r, cap=100000):
    """Add coverage for class [p, r] to cov array."""
    r_mod = r % p
    if r_mod == 0:
        start = p
    else:
        start = r_mod
    for n in range(start, cap + 1, p):
        cov[n] += 1

def _remove_class(cov, p, r, cap=100000):
    """Remove coverage for class [p, r] from cov array."""
    r_mod = r % p
    if r_mod == 0:
        start = p
    else:
        start = r_mod
    for n in range(start, cap + 1, p):
        cov[n] -= 1

def _compute_score_fast(classes, cap=100000):
    """Compute score using coverage array."""
    # Check budget
    budget = Fraction(0, 1)
    for p, r in classes:
        budget += Fraction(1, p)
    if budget > Fraction(1, 1):
        return 0
    cov = _build_coverage(classes, cap)
    return _compute_L_from_cov(cov, cap)

def _construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    primes = _sieve_primes(1000)
    cap = 100000
    
    # Strategy: Use a greedy approach to build a good covering
    # Start with the seed idea: primes from 13 upward, r = (i+1) % p
    # But we'll be smarter about which primes to use
    
    # Initial: use small primes for dense coverage
    # The key insight: small primes cover more numbers per unit of 1/p budget
    # So prioritize small primes
    
    # Start with a reasonable initial set
    # Use primes 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, ...
    # with r chosen to cover specific gaps
    
    # Let's start with the seed pattern but optimize
    init_primes = [p for p in primes if p >= 13]
    # Assign remainders: try to cover 1, 2, 3, ... 
    # For the i-th prime, set r so that it covers number i+1
    # But this may not be optimal; let's use a different approach
    
    # Better initial: for each prime p, set r = 0 (covers multiples of p)
    # Then adjust remainders to fill gaps
    
    # Actually, let's use the seed's initial and then optimize heavily
    init_classes = []
    budget = Fraction(0, 1)
    for i, p in enumerate(init_primes):
        r = (i + 1) % p
        if budget + Fraction(1, p) <= Fraction(1, 1):
            init_classes.append([p, r])
            budget += Fraction(1, p)
        else:
            break
    
    # Build initial coverage
    cov = [0] * (cap + 1)
    for p, r in init_classes:
        _add_class(cov, p, r, cap)
    
    current_L = _compute_L_from_cov(cov, cap)
    best_L = current_L
    best_classes = [list(c) for c in init_classes]
    
    # Local search: optimize remainders
    # For each prime, try all possible remainders and keep the best
    improved = True
    iteration = 0
    while improved and time.time() < deadline:
        improved = False
        iteration += 1
        
        for idx in range(len(best_classes)):
            if time.time() > deadline:
                break
            p, old_r = best_classes[idx]
            best_r = old_r
            best_local_L = best_L
            
            # Try all remainders for small primes, sample for large
            if p <= 50:
                candidates = list(range(p))
            else:
                # Try a strategic subset
                candidates = list(range(0, p, max(1, p // 30)))
                candidates = list(set(candidates + [0, 1, 2, 3, 4, 5]))
                candidates = [r for r in candidates if 0 <= r < p]
            
            for r in candidates:
                if r == old_r:
                    continue
                # Remove old, add new
                _remove_class(cov, p, old_r, cap)
                _add_class(cov, p, r, cap)
                new_L = _compute_L_from_cov(cov, cap)
                if new_L > best_local_L:
                    best_local_L = new_L
                    best_r = r
                else:
                    # Revert
                    _remove_class(cov, p, r, cap)
                    _add_class(cov, p, old_r, cap)
            
            if best_r != old_r:
                # Apply the best remainder
                _remove_class(cov, p, old_r, cap)
                _add_class(cov, p, best_r, cap)
                best_classes[idx] = [p, best_r]
                best_L = best_local_L
                improved = True
    
    # Try to improve by swapping primes
    if time.time() < deadline - 10:
        used_primes = set(p for p, r in best_classes)
        avail = [p for p in primes if p not in used_primes]
        
        # Try adding a new prime class
        for p in sorted(avail, key=lambda x: 1/x):  # Smallest 1/p first (largest p)
            if time.time() > deadline - 5:
                break
            budget = Fraction(0, 1)
            for pp, rr in best_classes:
                budget += Fraction(1, pp)
            if budget + Fraction(1, p) > Fraction(1, 1):
                continue
            
            # Try different remainders
            best_r = None
            best_local_L = best_L
            candidates = list(range(0, p, max(1, p // 20)))
            candidates = list(set(candidates + [0, 1, 2, 3]))
            candidates = [r for r in candidates if 0 <= r < p]
            
            for r in candidates:
                _add_class(cov, p, r, cap)
                new_L = _compute_L_from_cov(cov, cap)
                if new_L > best_local_L:
                    best_local_L = new_L
                    best_r = r
                else:
                    _remove_class(cov, p, r, cap)
            
            if best_r is not None:
                _add_class(cov, p, best_r, cap)
                best_classes.append([p, best_r])
                best_L = best_local_L
                used_primes.add(p)
                avail.remove(p)
    
    # Final local search
    if time.time() < deadline - 3:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for idx in range(len(best_classes)):
                if time.time() > deadline - 1:
                    break
                p, old_r = best_classes[idx]
                best_r = old_r
                best_local_L = best_L
                
                if p <= 30:
                    candidates = list(range(p))
                else:
                    candidates = list(range(0, p, max(1, p // 20)))
                    candidates = list(set(candidates + [0, 1, 2, 3, 4]))
                    candidates = [r for r in candidates if 0 <= r < p]
                
                for r in candidates:
                    if r == old_r:
                        continue
                    _remove_class(cov, p, old_r, cap)
                    _add_class(cov, p, r, cap)
                    new_L = _compute_L_from_cov(cov, cap)
                    if new_L > best_local_L:
                        best_local_L = new_L
                        best_r = r
                    else:
                        _remove_class(cov, p, r, cap)
                        _add_class(cov, p, old_r, cap)
                
                if best_r != old_r:
                    _remove_class(cov, p, old_r, cap)
                    _add_class(cov, p, best_r, cap)
                    best_classes[idx] = [p, best_r]
                    best_L = best_local_L
                    improved = True
    
    # Ensure validity
    final_classes = []
    seen_primes = set()
    final_budget = Fraction(0, 1)
    for p, r in best_classes:
        if p in seen_primes:
            continue
        r = r % p
        if 0 <= r < p:
            final_classes.append([p, r])
            seen_primes.add(p)
            final_budget += Fraction(1, p)
    
    while final_budget > Fraction(1, 1) and final_classes:
        final_classes.sort(key=lambda x: -x[0])
        p, r = final_classes.pop()
        final_budget -= Fraction(1, p)
    
    return final_classes

def construct():
    return _construct()
# EVOLVE-BLOCK-END
