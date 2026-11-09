"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 5000
_DEADLINE = time.time() + 310.0  # 330s wall-clock, 20s safety margin

# EVOLVE-BLOCK-START
def _primes(limit):
    """Return list of primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]


def construct():
    # Start from 1 and the primes above sqrt(N); repair any prime whose residue classes are all covered
    # by deleting the least-populated class, until the tuple is admissible.
    points = [1] + [p for p in _primes(N) if p * p > N]
    
    # Precompute primes up to N (max possible length)
    all_primes = _primes(N)
    
    # Use boolean array for active points for O(1) removal checks
    active = [True] * (N + 1)
    for v in points:
        active[v] = True
    
    changed = True
    while changed:
        if time.time() > _DEADLINE - 5.0:
            break
        changed = False
        L = len(points)
        # Only need to check primes <= L
        for p in all_primes:
            if p > L:
                break
            # Count residue classes
            class_counts = [0] * p
            for v in points:
                class_counts[v % p] += 1
            if min(class_counts) > 0:
                # All p residue classes are covered; remove the least-populated class
                min_count = min(class_counts)
                target_class = class_counts.index(min_count)
                # Remove all points in that residue class
                for v in points:
                    if v % p == target_class:
                        active[v] = False
                points = [v for v in points if active[v]]
                changed = True
                break
    
    return points
# EVOLVE-BLOCK-END
