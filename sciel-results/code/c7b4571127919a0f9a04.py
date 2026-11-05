"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 5000
DEADLINE = 320  # seconds, safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]


def _is_admissible(points, primes_cache):
    """Check if points is admissible: for every prime p <= len(points), some residue class mod p is missed."""
    n = len(points)
    for p in primes_cache:
        if p > n:
            break
        covered = set()
        for v in points:
            covered.add(v % p)
            if len(covered) == p:
                return False
    return True


def construct():
    start_time = time.time()
    
    # Precompute primes up to N+1
    max_len = N + 1
    primes_list = _primes(max_len)
    
    # Start with all numbers
    points = list(range(N + 1))
    point_set = set(points)
    
    # For each prime, track which residue classes are "protected" (must be missed)
    # We'll use a greedy approach: for each prime p <= |points|, if all residues covered,
    # remove the least populated residue class
    
    # To be efficient, we process primes in order and maintain counts
    # Key insight: once we remove a residue class mod p, we don't need to check p again
    # unless the set size changes such that p was previously > n but now p <= n (unlikely since n only decreases)
    
    changed = True
    while changed:
        if time.time() - start_time > DEADLINE:
            break
        
        changed = False
        n = len(points)
        
        for p in primes_list:
            if p > n:
                break
            
            # Count residue classes efficiently
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            
            # Check if all residue classes are covered
            if 0 not in counts:
                # All classes covered - need to remove one
                # Remove the least-populated class
                min_count = min(counts)
                smallest_class = counts.index(min_count)
                
                # Remove all elements in this residue class
                to_remove = [v for v in points if v % p == smallest_class]
                to_remove_set = set(to_remove)
                
                points = [v for v in points if v not in to_remove_set]
                changed = True
                break
    
    return points
# EVOLVE-BLOCK-END
