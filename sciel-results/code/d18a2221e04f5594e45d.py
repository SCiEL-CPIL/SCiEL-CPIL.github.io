"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 5000

# EVOLVE-BLOCK-START
def _primes_upto(limit):
    """Return list of all primes <= limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]


def _greedy_repair(points, all_primes, deadline):
    """
    Greedy repair: repeatedly find a prime p <= len(points) whose residue classes
    are all covered, and remove the least-populated class. Continue until admissible
    or deadline.
    """
    points = list(points)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        m = len(points)
        for p in all_primes:
            if p > m:
                break
            # Count residue classes
            class_counts = {}
            for v in points:
                r = v % p
                class_counts[r] = class_counts.get(r, 0) + 1
            if len(class_counts) == p:
                # Violation: all p classes covered
                min_count = min(class_counts.values())
                to_remove = set()
                for v in points:
                    if class_counts[v % p] == min_count:
                        to_remove.add(v)
                points = [v for v in points if v not in to_remove]
                changed = True
                break
    return points


def _check_admissible(points, all_primes):
    """Check if points is admissible for all primes p <= len(points)."""
    m = len(points)
    for p in all_primes:
        if p > m:
            break
        residues = set()
        for v in points:
            residues.add(v % p)
            if len(residues) == p:
                return False
    return True


def _local_improvement(points, all_primes, deadline):
    """
    Try to re-add removed elements or swap elements to increase tuple size
    while maintaining admissibility.
    """
    points_set = set(points)
    points = list(points)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = len(points)
        # Try adding each integer from 0 to N that is not in points
        for candidate in range(N + 1):
            if candidate in points_set:
                continue
            if time.time() >= deadline:
                break
            # Try adding candidate
            new_points = points + [candidate]
            new_m = len(new_points)
            # Check admissibility
            admissible = True
            for p in all_primes:
                if p > new_m:
                    break
                residues = set()
                for v in new_points:
                    residues.add(v % p)
                    if len(residues) == p:
                        admissible = False
                        break
            if admissible:
                points = new_points
                points_set.add(candidate)
                improved = True
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s kill

    # Precompute primes up to N+1
    all_primes = _primes_upto(N + 1)

    # Strategy 1: Start from a sparse set (1 and primes > sqrt(N)) and repair
    # This tends to give a larger initial set than starting from all integers
    points_sparse = [1] + [p for p in all_primes if p * p > N]
    points_sparse = _greedy_repair(points_sparse, all_primes, deadline)

    # Strategy 2: Start from all integers and repair
    points_dense = list(range(N + 1))
    points_dense = _greedy_repair(points_dense, all_primes, deadline)

    # Pick the better one
    points = points_sparse if len(points_sparse) >= len(points_dense) else points_dense

    # Local improvement: try to add back elements
    if time.time() < deadline:
        points = _local_improvement(points, all_primes, deadline)

    # Final verification and repair if needed
    if not _check_admissible(points, all_primes):
        points = _greedy_repair(points, all_primes, deadline)

    return points
# EVOLVE-BLOCK-END
