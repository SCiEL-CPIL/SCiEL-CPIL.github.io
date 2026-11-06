"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 5000
_DEADLINE = 300.0  # safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def construct():
    """Start from full set [0, N], greedily remove the globally least-populated
    residue class among all violating primes until admissible.
    
    Each iteration: find all primes p <= len(points) where all p residue classes
    are covered. Among those, pick the (p, r) pair with minimum count of points
    in class r mod p. Remove all points in that class. Repeat until admissible.
    """
    start_time = time.time()
    
    points = list(range(N + 1))
    all_primes = _primes(N)
    
    while True:
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        current_len = len(points)
        if current_len <= 1:
            break
        
        # Find all primes p <= current_len
        relevant_primes = [p for p in all_primes if p <= current_len]
        if not relevant_primes:
            break
        
        # For each relevant prime, count residue classes and check if all covered
        # Track the minimum-count class among all violating primes
        best_p = None
        best_r = None
        best_count = float('inf')
        
        for p in relevant_primes:
            # Count residues using numpy for speed
            arr = np.array(points)
            residues = arr % p
            counts = np.bincount(residues, minlength=p)
            
            # Check if all residue classes are covered
            if counts.min() > 0:
                # All classes covered - this prime is violated
                min_r = int(np.argmin(counts))
                min_c = int(counts[min_r])
                if min_c < best_count:
                    best_count = min_c
                    best_p = p
                    best_r = min_r
        
        if best_p is None:
            # No violating primes - tuple is admissible
            break
        
        # Remove all points with residue best_r mod best_p
        points = [v for v in points if v % best_p != best_r]
    
    return points
# EVOLVE-BLOCK-END
