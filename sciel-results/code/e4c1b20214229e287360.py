"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 5000

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


def _is_admissible(points, primes_needed):
    """Check if points is admissible w.r.t. all primes in primes_needed."""
    for p in primes_needed:
        residues = set()
        for v in points:
            residues.add(v % p)
        if len(residues) == p:
            return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 280  # safety margin before 300s kill
    
    # Start with all integers in [0, N]
    points = list(range(N + 1))
    
    # We need admissibility for all primes p <= len(points)
    # As we remove elements, len(points) shrinks, so we may need fewer primes
    # But we must ensure admissibility for all primes up to current len
    
    # Strategy: repeatedly find a prime p <= len(points) where all p residue classes
    # are covered, and remove the smallest residue class. Prioritize primes where
    # removing the smallest class causes minimal damage (smallest class size).
    
    while time.time() < deadline:
        k = len(points)
        if k == 0:
            break
        
        # Get primes up to k
        primes_up_to_k = _primes(k)
        
        # Find all "bad" primes (where all residue classes are covered)
        bad_primes = []
        for p in primes_up_to_k:
            residues = set()
            for v in points:
                residues.add(v % p)
            if len(residues) == p:
                # Find the smallest residue class
                class_sizes = {}
                for v in points:
                    r = v % p
                    class_sizes[r] = class_sizes.get(r, 0) + 1
                min_size = min(class_sizes.values())
                bad_primes.append((min_size, p))
        
        if not bad_primes:
            # Admissible!
            break
        
        # Sort by min class size (remove smallest class first to preserve more elements)
        bad_primes.sort()
        min_size, p = bad_primes[0]
        
        # Find all residue classes with the minimum size
        class_sizes = {}
        for v in points:
            r = v % p
            class_sizes[r] = class_sizes.get(r, 0) + 1
        
        min_class = min(r for r, s in class_sizes.items() if s == min_size)
        
        # Remove all elements in that residue class
        points = [v for v in points if v % p != min_class]
    
    # Final check: ensure admissibility
    k = len(points)
    primes_up_to_k = _primes(k)
    for p in primes_up_to_k:
        residues = set()
        for v in points:
            residues.add(v % p)
        if len(residues) == p:
            # Shouldn't happen, but repair if it does
            while len(residues) == p:
                class_sizes = {}
                for v in points:
                    r = v % p
                    class_sizes[r] = class_sizes.get(r, 0) + 1
                min_size = min(class_sizes.values())
                min_class = min(r for r, s in class_sizes.items() if s == min_size)
                points = [v for v in points if v % p != min_class]
                k = len(points)
                primes_up_to_k = _primes(k)
                residues = set()
                for v in points:
                    residues.add(v % p)
    
    return points
# EVOLVE-BLOCK-END
