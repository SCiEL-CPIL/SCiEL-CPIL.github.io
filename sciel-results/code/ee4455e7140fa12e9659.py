"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    """Sieve of Eratosthenes to find all primes up to limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _count_residues(points, p):
    """Count occurrences of each residue class mod p among points."""
    counts = [0] * p
    for v in points:
        counts[v % p] += 1
    return counts


def construct():
    deadline = time.time() + 280  # safety margin before 330s kill
    
    # Start with all integers [0, N]
    points = list(range(N + 1))
    
    # Precompute primes up to N+1 (max possible tuple size)
    all_primes = _primes(N + 1)
    
    # Iteratively repair: for each prime p <= len(points), ensure not all residues are present
    # Process primes in ascending order (smaller primes are more constraining)
    changed = True
    while changed:
        changed = False
        current_len = len(points)
        
        # Get primes up to current_len
        relevant_primes = [p for p in all_primes if p <= current_len]
        
        for p in relevant_primes:
            if time.time() > deadline:
                break
            counts = _count_residues(points, p)
            # Check if all residue classes are present
            if min(counts) > 0:
                # Find the least-populated residue class
                min_count = min(counts)
                # Remove all points in the least-populated class(es)
                # Choose the class with minimum count; if tie, pick the one
                # whose removal hurts larger primes least (heuristic: just pick min)
                target_class = counts.index(min_count)
                points = [v for v in points if v % p != target_class]
                changed = True
                break  # restart the prime loop since set changed
        
        if time.time() > deadline:
            break
    
    # Final verification pass: ensure admissibility
    # For every prime p <= len(points), some residue class mod p is missed
    final_len = len(points)
    for p in all_primes:
        if p > final_len:
            break
        counts = _count_residues(points, p)
        if min(counts) > 0:
            # This shouldn't happen if the loop converged, but fix if it does
            min_count = min(counts)
            target_class = counts.index(min_count)
            points = [v for v in points if v % p != target_class]
    
    return points
# EVOLVE-BLOCK-END
