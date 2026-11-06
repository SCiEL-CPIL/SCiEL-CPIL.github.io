"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from fractions import Fraction

# Precompute primes up to 1000
def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

_PRIMES = _primes_upto(1000)
# Map prime to index for quick lookup
_PRIME_SET = set(_PRIMES)

def _reciprocal_sum(classes):
    """Compute sum of 1/p for classes [[p, r], ...] as a Fraction."""
    s = Fraction(0, 1)
    for p, r in classes:
        s += Fraction(1, p)
    return s

def _coverage(classes, limit=100000):
    """Return the largest L such that every integer 1..L is covered by some class."""
    if not classes:
        return 0
    # Build a boolean array for coverage up to limit
    covered = [False] * (limit + 1)
    for p, r in classes:
        # r is the residue: numbers n such that n % p == r are covered
        # We need to cover 1..limit
        if r == 0:
            start = p
        else:
            start = r
        # But we only care about 1..limit
        if start < 1:
            start = 1
        if start > limit:
            continue
        for n in range(start, limit + 1, p):
            covered[n] = True
    # Find the largest L such that covered[1..L] are all True
    L = 0
    for i in range(1, limit + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Strategy: Start with the seed solution, then try to improve coverage
    # The seed covers primes from 13 upward, mapping i-th prime to residue (i+1) % p
    # This gives sum of reciprocals ~ 0.92 and some coverage
    
    # Better strategy: Use a greedy/local search approach
    # 1. Start with a set of classes that covers a good range
    # 2. Try to add classes (with small reciprocals) to extend coverage
    # 3. Try to swap classes to improve coverage while maintaining sum <= 1
    
    # First, let's try a more systematic approach:
    # Use smaller primes with carefully chosen residues to maximize coverage
    
    # The key insight: small primes cover many numbers in [1, L]
    # But we have a budget of sum(1/p) <= 1
    
    # Let's try a greedy approach:
    # 1. Sort primes by size (smaller primes first, as they cover more)
    # 2. For each prime, try all residues and pick the one that extends coverage most
    # 3. Continue until budget is exhausted
    
    # However, this is complex. Let's start with the seed and try local search.
    
    # Seed solution:
    primes_seed = [p for p in _PRIMES if p >= 13]
    classes_seed = [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
    budget_seed = _reciprocal_sum(classes_seed)
    L_seed = _coverage(classes_seed, 2000)
    
    best_classes = classes_seed
    best_L = L_seed
    best_budget = budget_seed
    
    # Try to improve by swapping residues
    # For each class, try changing its residue to see if it improves coverage
    for iteration in range(50):
        if time.time() > deadline - 5:
            break
        improved = False
        for idx, (p, r) in enumerate(best_classes):
            if time.time() > deadline - 5:
                break
            # Try all residues for this prime
            for new_r in range(p):
                if new_r == r:
                    continue
                test_classes = best_classes.copy()
                test_classes[idx] = [p, new_r]
                test_L = _coverage(test_classes, 2000)
                if test_L > best_L:
                    best_classes = test_classes
                    best_L = test_L
                    improved = True
                    break
        if not improved:
            break
    
    # Try adding more classes if we have budget
    used_primes = set(p for p, r in best_classes)
    current_budget = _reciprocal_sum(best_classes)
    
    # Sort unused primes by size (smaller first, as they cover more per unit budget)
    unused_primes = sorted([p for p in _PRIMES if p not in used_primes], key=lambda x: x)
    
    for p in unused_primes:
        if time.time() > deadline - 5:
            break
        cost = Fraction(1, p)
        if current_budget + cost > Fraction(1, 1):
            continue
        # Try all residues for this prime
        for new_r in range(p):
            test_classes = best_classes + [[p, new_r]]
            test_L = _coverage(test_classes, 2000)
            if test_L > best_L:
                best_classes = test_classes
                best_L = test_L
                current_budget += cost
                break
    
    # Final check
    final_budget = _reciprocal_sum(best_classes)
    if final_budget > Fraction(1, 1):
        # Fallback to seed if we somehow exceeded budget
        return classes_seed
    
    return best_classes
# EVOLVE-BLOCK-END
