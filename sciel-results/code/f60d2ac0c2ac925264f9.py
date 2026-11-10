"""Combined program: admissible-tuple maximization in [0, 5000].

Uses a vectorized NumPy-based tracker for fast residue-class accounting,
greedy removal for feasibility, and a multi-phase add-back with local
swap-based polishing to push the tuple size as high as possible.
"""
import time
import random
import numpy as np

N = 5000
_DEADLINE = 295.0  # safety margin before 330s kill


def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return [int(p) for p in np.where(sieve)[0]]


class _Tracker:
    """Vectorized residue-class tracker using NumPy."""

    def __init__(self, primes):
        self.primes = primes
        self.p_arr = np.array(primes, dtype=np.int64)
        self.residue_counts = {}
        self.nonzero = {}
        for p in primes:
            self.residue_counts[p] = np.zeros(p, dtype=np.int32)
            self.nonzero[p] = 0

    def rebuild(self, points):
        """Rebuild from a list/set of points."""
        pts = np.array(list(points), dtype=np.int64) if len(points) else np.array([], dtype=np.int64)
        for p in self.primes:
            self.residue_counts[p] = np.zeros(p, dtype=np.int32)
            self.nonzero[p] = 0
        if len(pts) == 0:
            return
        for p in self.primes:
            residues = pts % p
            counts = np.bincount(residues, minlength=p).astype(np.int32)
            self.residue_counts[p] = counts
            self.nonzero[p] = int(np.sum(counts > 0))

    def covered_primes(self, max_p=None):
        """Return list of primes where all residue classes are covered."""
        result = []
        limit = max_p if max_p is not None else N + 1
        for p in self.primes:
            if p > limit:
                break
            if self.nonzero[p] == p:
                result.append(p)
        return result

    def can_add(self, v, max_p=None):
        """Check if v can be added without violating admissibility."""
        limit = max_p if max_p is not None else N + 1
        for p in self.primes:
            if p > limit:
                break
            r = v % p
            if self.residue_counts[p][r] == 0 and self.nonzero[p] == p - 1:
                return False
        return True

    def add(self, v):
        """Add point v, updating all prime residue counts."""
        for p in self.primes:
            r = v % p
            if self.residue_counts[p][r] == 0:
                self.nonzero[p] += 1
            self.residue_counts[p][r] += 1

    def remove(self, v):
        """Remove point v, updating all prime residue counts."""
        for p in self.primes:
            r = v % p
            self.residue_counts[p][r] -= 1
            if self.residue_counts[p][r] == 0:
                self.nonzero[p] -= 1

    def swap(self, v_out, v_in):
        """Swap: remove v_out, add v_in."""
        self.remove(v_out)
        self.add(v_in)

    def risk_score(self, v, max_p=None):
        """Count primes where v fills a new (currently empty) class."""
        limit = max_p if max_p is not None else N + 1
        score = 0
        for p in self.primes:
            if p > limit:
                break
            r = v % p
            if self.residue_counts[p][r] == 0:
                score += 1
        return score


def _greedy_removal(points, tracker, deadline):
    """Greedy removal: find covered prime, remove least-populated class."""
    point_set = set(points)
    tracker.rebuild(point_set)

    while time.time() < deadline:
        covered = tracker.covered_primes()
        if not covered:
            break

        p = min(covered)
        counts = tracker.residue_counts[p]
        # Find residue class with minimum count
        min_class = int(np.argmin(counts))

        to_remove = [v for v in point_set if v % p == min_class]
        for v in to_remove:
            point_set.discard(v)
            tracker.remove(v)

    return point_set


def _add_back(points, tracker, deadline):
    """Add-back search: repeatedly add lowest-risk candidates."""
    point_set = set(points)
    tracker.rebuild(point_set)
    all_candidates = set(range(N + 1)) - point_set

    improved = True
    while improved and time.time() < deadline:
        improved = False
        if not all_candidates:
            break

        # Compute risk scores for all candidates
        candidates = list(all_candidates)
        scores = [(tracker.risk_score(v), v) for v in candidates]
        scores.sort()

        for score, v in scores:
            if time.time() >= deadline:
                break
            if tracker.can_add(v):
                tracker.add(v)
                point_set.add(v)
                all_candidates.discard(v)
                improved = True
                break  # Restart with fresh scores

    return point_set


def _local_swap_polish(points, tracker, deadline):
    """Local search: try swapping one included point with one excluded to improve."""
    point_set = set(points)
    tracker.rebuild(point_set)
    excluded = set(range(N + 1)) - point_set

    improved = True
    while improved and time.time() < deadline:
        improved = False
        if not excluded or not point_set:
            break

        # Try random swaps
        for _ in range(200):
            if time.time() >= deadline:
                break
            v_out = random.choice(list(point_set))
            v_in = random.choice(list(excluded))
            if v_in in point_set:
                continue

            # Check if swap maintains admissibility
            # Temporarily remove v_out
            tracker.remove(v_out)
            # Check if v_in can be added
            ok = tracker.can_add(v_in)
            if ok:
                tracker.add(v_in)
                point_set.discard(v_out)
                point_set.add(v_in)
                excluded.discard(v_in)
                excluded.add(v_out)
                improved = True
                break
            else:
                # Restore v_out
                tracker.add(v_out)

    return point_set


def _full_pipeline(points, tracker, deadline):
    """Run greedy removal, add-back, and local swap polish."""
    point_set = _greedy_removal(points, tracker, deadline)
    point_set = _add_back(point_set, tracker, deadline)
    point_set = _local_swap_polish(point_set, tracker, deadline)
    return point_set


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE

    all_primes = _primes(N + 1)
    tracker = _Tracker(all_primes)
    best = set()

    # Strategy 1: Start from full set [0, N] and sieve down
    if time.time() < deadline - 30:
        init_full = list(range(N + 1))
        result = _full_pipeline(init_full, tracker, deadline)
        if len(result) > len(best):
            best = result

    # Strategy 2: Random restarts with different densities
    attempts = 0
    while time.time() < deadline - 30 and attempts < 6:
        fraction = 0.25 + 0.5 * (attempts % 5) / 5.0
        seed = attempts * 7919 + 12345
        random.seed(seed)
        init = random.sample(range(N + 1), int(fraction * (N + 1)))
        result = _full_pipeline(init, tracker, deadline)
        if len(result) > len(best):
            best = result
        attempts += 1

    # Final polish: additional add-back and swap on the best
    if time.time() < deadline - 5 and best:
        best = _add_back(best, tracker, deadline)
        best = _local_swap_polish(best, tracker, deadline)

    return list(best)
