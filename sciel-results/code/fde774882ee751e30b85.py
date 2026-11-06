"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _primes_up_to(n):
    return [p for p in range(2, n + 1) if _is_prime(p)]

def _compute_coverage_fast(classes, limit=10000):
    """Compute the largest L such that every integer 1..L is covered using bitmasks."""
    if not classes:
        return 0
    # Use a bytearray for coverage
    covered = bytearray(limit + 1)
    for p, r in classes:
        # Mark all x ≡ r (mod p) in [1, limit]
        # Start from max(1, r) but r can be 0, so we need to handle that
        # If r == 0, the first positive number is p (if p <= limit)
        if r == 0:
            start = p if p <= limit else limit + 1
        else:
            start = r if r >= 1 else p
        for x in range(start, limit + 1, p):
            if x >= 1:
                covered[x] = 1
    L = 0
    for i in range(1, limit + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def _budget_check(classes):
    """Check if sum of 1/p <= 1 using exact fractions."""
    total = Fraction(0)
    for p, r in classes:
        total += Fraction(1, p)
    return total <= 1

def _budget_fraction(classes):
    """Return the sum of 1/p as a Fraction."""
    total = Fraction(0)
    for p, r in classes:
        total += Fraction(1, p)
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    all_primes = _primes_up_to(1000)
    
    # We need a valid initial set.
    # Let's try a set that is known to be valid and covers well.
    # Based on analysis, {2, 5, 7, 11, 17} is valid (sum ~0.99).
    # Let's start with this and optimize residues.
    
    initial_primes = [2, 5, 7, 11, 17]
    initial_classes = [[p, 0] for p in initial_primes]
    
    # Verify budget
    if not _budget_check(initial_classes):
        # Fallback to a smaller set
        initial_primes = [2, 3, 7, 37, 337]
        initial_classes = [[p, 0] for p in initial_primes]
        if not _budget_check(initial_classes):
            initial_primes = [2, 3, 7, 37]
            initial_classes = [[p, 0] for p in initial_primes]
            if not _budget_check(initial_classes):
                initial_primes = [2, 3, 7]
                initial_classes = [[p, 0] for p in initial_primes]
                if not _budget_check(initial_classes):
                    initial_primes = [2, 3]
                    initial_classes = [[p, 0] for p in initial_primes]
                    if not _budget_check(initial_classes):
                        initial_primes = [2]
                        initial_classes = [[p, 0] for p in initial_primes]
    
    best_classes = [list(c) for c in initial_classes]
    best_L = _compute_coverage_fast(best_classes, limit=10000)
    
    # Local search: Optimize residues
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for idx in range(len(best_classes)):
            p, r = best_classes[idx]
            best_r = r
            best_L_local = best_L
            
            for trial_r in range(p):
                if trial_r == r:
                    continue
                old_r = best_classes[idx][1]
                best_classes[idx][1] = trial_r
                L = _compute_coverage_fast(best_classes, limit=10000)
                if L > best_L_local:
                    best_L_local = L
                    best_r = trial_r
                else:
                    best_classes[idx][1] = old_r
            
            if best_r != r:
                best_classes[idx][1] = best_r
                best_L = best_L_local
                improved = True
    
    # Try to add more primes if budget allows
    used_primes = set(p for p, r in best_classes)
    available_primes = [p for p in all_primes if p not in used_primes]
    # Sort by 1/p ascending (cheapest first) to allow more primes
    available_primes.sort(key=lambda p: 1.0/p)
    
    for p in available_primes:
        if time.time() > deadline:
            break
        
        test_classes = [list(c) for c in best_classes] + [[p, 0]]
        if not _budget_check(test_classes):
            continue
        
        best_r = None
        best_L_local = best_L
        for trial_r in range(p):
            test_classes[-1][1] = trial_r
            L = _compute_coverage_fast(test_classes, limit=10000)
            if L > best_L_local:
                best_L_local = L
                best_r = trial_r
        
        if best_r is not None and best_L_local > best_L:
            best_classes.append([p, best_r])
            used_primes.add(p)
            best_L = best_L_local
    
    # Final local search
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for idx in range(len(best_classes)):
            p, r = best_classes[idx]
            best_r = r
            best_L_local = best_L
            
            for trial_r in range(p):
                if trial_r == r:
                    continue
                old_r = best_classes[idx][1]
                best_classes[idx][1] = trial_r
                L = _compute_coverage_fast(best_classes, limit=10000)
                if L > best_L_local:
                    best_L_local = L
                    best_r = trial_r
                else:
                    best_classes[idx][1] = old_r
            
            if best_r != r:
                best_classes[idx][1] = best_r
                best_L = best_L_local
                improved = True
    
    # Final coverage computation
    final_L = _compute_coverage_fast(best_classes, limit=100000)
    
    return best_classes
# EVOLVE-BLOCK-END
