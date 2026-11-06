import math
import time
import random
from fractions import Fraction

# Precompute primes up to 1000
def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

PRIMES = _sieve(1000)
PRIME_SET = set(PRIMES)

def _compute_coverage(classes, cap=100000):
    """Compute the largest L such that every integer 1..L is covered."""
    if not classes:
        return 0
    # Use a bytearray for efficiency
    covered = bytearray(cap + 1)  # indices 1..cap
    for p, r in classes:
        # Mark all numbers n in [1, cap] with n % p == r
        # Start from the first number >= 1 with n % p == r
        if r == 0:
            start = p
        else:
            start = r if r >= 1 else r + p
        if start < 1:
            start += p * ((1 - start + p - 1) // p)
        # Actually simpler: find smallest n >= 1 with n % p == r
        # n = r + k*p, need n >= 1
        # if r >= 1: n = r
        # if r == 0: n = p
        # if r < 1 (not possible since 0 <= r < p): n = r + p
        if r >= 1:
            start = r
        else:
            start = r + p
        for n in range(start, cap + 1, p):
            covered[n] = 1
    # Find largest L such that all 1..L are covered
    L = 0
    for i in range(1, cap + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def _sum_reciprocals(classes):
    """Compute sum of 1/p as Fraction."""
    s = Fraction(0, 1)
    for p, r in classes:
        s += Fraction(1, p)
    return s

def _is_valid(classes):
    """Check distinct primes, 0 <= r < p, sum 1/p <= 1."""
    seen = set()
    s = Fraction(0, 1)
    for p, r in classes:
        if p in seen:
            return False
        if p < 2 or p > 1000 or not PRIME_SET.__contains__(p):
            return False
        if r < 0 or r >= p:
            return False
        seen.add(p)
        s += Fraction(1, p)
        if s > 1:
            return False
    return True

def _greedy_init():
    """Greedy initialization: pick primes that maximize coverage per unit cost."""
    classes = []
    used_primes = set()
    budget = Fraction(1, 1)
    
    # Start with a simple heuristic: try to cover 1, 2, 3, ...
    # For each uncovered number, try to find a prime class that covers it and more
    # But this is slow. Instead, use a randomized greedy approach.
    
    # Sort primes by some heuristic
    # Let's try: for each prime, compute how many numbers in 1..2000 it can cover
    # and pick the best ones first
    
    # Actually, let's use a different approach:
    # Start with empty set, repeatedly add the prime class that gives best marginal coverage
    # This is expensive. Let's use a simpler seed and improve via local search.
    
    # Seed: use primes starting from some value
    # The seed program uses primes from 13 upward, covering i with the i-th prime
    # Let's start with a similar approach but optimized
    
    # Actually, let's just use a random initialization and improve
    return classes

def _local_search(classes, max_time=100.0):
    """Local search to improve coverage."""
    start_time = time.time()
    best_classes = list(classes)
    best_coverage = _compute_coverage(best_classes)
    best_sum = _sum_reciprocals(best_classes)
    
    # Precompute coverage for current solution
    def get_coverage(classes):
        return _compute_coverage(classes)
    
    # Local search: try changing residues, adding/removing primes
    # Use simulated annealing or hill climbing
    
    for iteration in range(100000):
        if time.time() - start_time > max_time:
            break
        
        # Randomly pick a move type
        move_type = random.random()
        
        if move_type < 0.4:
            # Change residue of an existing class
            if not best_classes:
                continue
            idx = random.randint(0, len(best_classes) - 1)
            p, r = best_classes[idx]
            new_r = random.randint(0, p - 1)
            if new_r == r:
                continue
            new_classes = list(best_classes)
            new_classes[idx] = [p, new_r]
            cov = get_coverage(new_classes)
            if cov > best_coverage:
                best_classes = new_classes
                best_coverage = cov
                best_sum = _sum_reciprocals(best_classes)
        
        elif move_type < 0.7:
            # Add a new prime class
            available = [p for p in PRIMES if p not in {c[0] for c in best_classes}]
            if not available:
                continue
            p = random.choice(available)
            # Check if we can afford it
            if best_sum + Fraction(1, p) > 1:
                continue
            r = random.randint(0, p - 1)
            new_classes = best_classes + [[p, r]]
            cov = get_coverage(new_classes)
            if cov > best_coverage:
                best_classes = new_classes
                best_coverage = cov
                best_sum = _sum_reciprocals(best_classes)
        
        elif move_type < 0.85:
            # Remove a prime class
            if len(best_classes) <= 1:
                continue
            idx = random.randint(0, len(best_classes) - 1)
            new_classes = [c for i, c in enumerate(best_classes) if i != idx]
            cov = get_coverage(new_classes)
            if cov > best_coverage:
                best_classes = new_classes
                best_coverage = cov
                best_sum = _sum_reciprocals(best_classes)
        
        else:
            # Swap: remove one, add another
            if not best_classes:
                continue
            remove_idx = random.randint(0, len(best_classes) - 1)
            removed_p = best_classes[remove_idx][0]
            available = [p for p in PRIMES if p not in {c[0] for c in best_classes}]
            if not available:
                continue
            add_p = random.choice(available)
            # Check budget: removing removed_p frees 1/removed_p, adding add_p costs 1/add_p
            if best_sum - Fraction(1, removed_p) + Fraction(1, add_p) > 1:
                continue
            add_r = random.randint(0, add_p - 1)
            new_classes = [c for i, c in enumerate(best_classes) if i != remove_idx]
            new_classes.append([add_p, add_r])
            cov = get_coverage(new_classes)
            if cov > best_coverage:
                best_classes = new_classes
                best_coverage = cov
                best_sum = _sum_reciprocals(best_classes)
    
    return best_classes, best_coverage

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial seed: use the approach from the seed program
    # Primes from 13 upward, covering consecutive integers
    primes_list = [p for p in PRIMES if p >= 13]
    classes = []
    s = Fraction(0, 1)
    for i, p in enumerate(primes_list):
        r = (i + 1) % p
        if s + Fraction(1, p) > 1:
            break
        classes.append([p, r])
        s += Fraction(1, p)
    
    current_coverage = _compute_coverage(classes)
    
    # Local search with time guard
    classes, current_coverage = _local_search(classes, max_time=100.0)
    
    # Final check
    if not _is_valid(classes):
        # Fallback to seed
        classes = []
        s = Fraction(0, 1)
        for i, p in enumerate(primes_list):
            r = (i + 1) % p
            if s + Fraction(1, p) > 1:
                break
            classes.append([p, r])
            s += Fraction(1, p)
    
    return classes

# EVOLVE-BLOCK-END
