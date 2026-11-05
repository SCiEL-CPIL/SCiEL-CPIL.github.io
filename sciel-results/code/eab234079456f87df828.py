"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 5000
DEADLINE = time.time() + 280  # safety margin before 300s kill

# EVOLVE-BLOCK-START

def _primes(limit):
    """Sieve of Eratosthenes to find all primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]


def construct():
    """
    Construct the largest admissible tuple in [0, N].
    
    Strategy: Start with all integers [0, N]. For each prime p <= len(current_set),
    if all p residue classes mod p are present, remove the least-populated class.
    Repeat until admissible or time limit.
    """
    start_time = time.time()
    
    # Start with all numbers
    points = list(range(N + 1))
    current_size = len(points)
    
    # Precompute primes up to N+1 (max possible size)
    max_prime_limit = N + 1
    primes = _primes(max_prime_limit)
    
    # We need to check primes p <= len(points)
    # Start with the full set and iteratively fix violations
    
    # For efficiency, track residue class counts
    # We'll iterate: for each prime p <= current_size, check if all classes present
    # If so, remove the smallest class
    
    # Use a set for O(1) membership and a list for iteration
    point_set = set(points)
    
    # We'll do multiple passes until admissible
    # In each pass, process primes in order and fix violations
    # To avoid infinite loops, we only remove elements (monotonic)
    
    admissible = False
    while not admissible:
        if time.time() - start_time > DEADLINE:
            break
        
        admissible = True
        current_size = len(point_set)
        
        # Find all primes <= current_size
        relevant_primes = [p for p in primes if p <= current_size]
        
        for p in relevant_primes:
            if time.time() - start_time > DEADLINE:
                break
            
            # Count residue classes
            class_counts = {}
            for v in point_set:
                r = v % p
                class_counts[r] = class_counts.get(r, 0) + 1
            
            if len(class_counts) == p:
                # All p classes are present - violation
                # Find the class with minimum count
                min_class = min(class_counts, key=lambda r: class_counts[r])
                # Remove all elements in that class
                to_remove = [v for v in point_set if v % p == min_class]
                for v in to_remove:
                    point_set.discard(v)
                admissible = False
                # After removing, sizes change, so restart prime check
                break
        
        if admissible:
            break
    
    return list(point_set)

# EVOLVE-BLOCK-END
