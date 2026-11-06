"""Combined program: seed repair-loop baseline + inspiration set-cover/local-improvement.

Strategy:
  * Baseline (seed): start from [0, N], repair by deleting the least-populated
    residue class for each prime p <= len(tuple) whose classes are all covered.
    This is the fixed-diameter greedy deletion and is a strong baseline.
  * Refinement (inspiration): treat the problem as set-cover over the primes
    that are "full" in [0, N], then run a local-improvement pass that re-adds
    removed elements (and swaps) to push the size up, all under a wall-clock
    deadline with a safety margin.
"""
import time
import sys
import numpy as np

N = 5000
DEADLINE_MARGIN = 30.0
T_START = time.time()


def _time_left():
    return (330.0 - DEADLINE_MARGIN) - (time.time() - T_START)


def _primes(limit):
    """Sieve of Eratosthenes: list of primes <= limit."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [i for i in range(2, limit + 1) if sieve[i]]


class FastAdmissibleChecker:
    """Fast admissibility checker using NumPy for vectorized residue class coverage."""
    
    def __init__(self, n_points):
        self.n_points = n_points
        self.primes = _primes(n_points)
        # Precompute bitmasks for each prime
        self.prime_masks = {}
        for p in self.primes:
            self.prime_masks[p] = (1 << p) - 1
    
    def check(self, points):
        """Check if points is admissible using NumPy operations."""
        m = len(points)
        if m == 0:
            return True
        
        if not isinstance(points, np.ndarray):
            points_arr = np.array(points, dtype=np.int64)
        else:
            points_arr = points
        
        # Get primes up to m
        primes_to_check = [p for p in self.primes if p <= m]
        
        for p in primes_to_check:
            # Compute residues
            residues = points_arr % p
            # Count unique residues
            unique_count = len(np.unique(residues))
            if unique_count == p:
                return False
        return True


def _check_admissible(points):
    """Return True iff `points` (distinct ints) is an admissible tuple."""
    if not points:
        return True
    m = len(points)
    checker = FastAdmissibleChecker(m)
    return checker.check(points)


def _greedy_repair(points):
    """Seed repair: delete least-populated class for each violating prime."""
    points = np.array(points, dtype=np.int64)
    changed = True
    while changed:
        changed = False
        m = len(points)
        primes_to_check = _primes(m)
        for p in primes_to_check:
            # Compute residues
            residues = points % p
            # Count occurrences of each residue
            counts = np.bincount(residues, minlength=p)
            if len(np.nonzero(counts)[0]) == p:
                # Find the residue class with the smallest count
                min_count_idx = np.argmin(counts)
                # Remove all points with this residue
                mask = residues != min_count_idx
                points = points[mask]
                changed = True
                break
    return points.tolist()


def _local_improve(points, removed):
    """Try to re-add removed elements and perform 1-for-1 swaps to grow size."""
    best = np.array(points, dtype=np.int64)
    best_set = set(best.tolist())
    removed_list = list(removed)
    removed_set = set(removed_list)
    
    # Precompute checker for current size
    max_possible_size = len(best) + len(removed)
    checker = FastAdmissibleChecker(max_possible_size)
    
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        # Re-add pass: try to add back removed elements
        for x in sorted(removed_set):
            if _time_left() < 5.0:
                break
            trial = np.concatenate([best, [x]])
            if checker.check(trial):
                best = trial
                best_set.add(x)
                removed_set.discard(x)
                improved = True
        # Swap pass: remove one kept, add one removed
        if _time_left() < 5.0:
            break
        best_list = best.tolist()
        for x in sorted(removed_set):
            if _time_left() < 5.0:
                break
            for y in best_list:
                if _time_left() < 5.0:
                    break
                if y == x:
                    continue
                trial_list = [v for v in best_list if v != y] + [x]
                trial = np.array(trial_list, dtype=np.int64)
                if checker.check(trial):
                    best = trial
                    best_set.discard(y)
                    best_set.add(x)
                    removed_set.discard(x)
                    removed_set.add(y)
                    best_list = trial.tolist()
                    improved = True
                    break
    return best.tolist()


def construct():
    # ---------- Baseline: seed repair on [0, N] ----------
    points = list(range(N + 1))
    points = _greedy_repair(points)
    best = points if _check_admissible(points) else []
    removed = [x for x in range(N + 1) if x not in set(best)]

    # ---------- Refinement: local improvement ----------
    try:
        if _time_left() > 5.0:
            improved = _local_improve(best, removed)
            if _check_admissible(improved) and len(improved) > len(best):
                best = improved
    except Exception:
        pass

    # ---------- Second pass: try to recover more with remaining time ----------
    try:
        if _time_left() > 10.0:
            current_best_set = set(best)
            still_removed = [x for x in range(N + 1) if x not in current_best_set]
            if still_removed:
                improved2 = _local_improve(best, still_removed)
                if _check_admissible(improved2) and len(improved2) > len(best):
                    best = improved2
    except Exception:
        pass

    # ---------- Final safety: ensure admissible, fall back to baseline ----------
    if not _check_admissible(best):
        fallback = _greedy_repair(list(range(N + 1)))
        best = fallback if _check_admissible(fallback) else []

    return list(best)
