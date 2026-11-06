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


# Precompute primes up to N once; admissibility only needs primes <= len(points) <= N+1.
_PRIMES_UP_TO_N = _primes(N + 1)


def _check_admissible(points):
    """Fast admissibility check via NumPy bincount.

    A tuple of size m is admissible iff for every prime p <= m,
    not all p residue classes mod p are represented.
    """
    if not points:
        return True
    m = len(points)
    arr = np.asarray(points, dtype=np.int64)
    for p in _PRIMES_UP_TO_N:
        if p > m:
            break
        # Count occurrences of each residue class mod p.
        # bincount with minlength=p gives a length-p array.
        counts = np.bincount(arr % p, minlength=p)
        if np.all(counts > 0):
            return False
    return True


def _greedy_repair(points):
    """Seed repair: delete least-populated class for each violating prime."""
    changed = True
    while changed:
        changed = False
        for p in _primes(len(points)):
            classes = {}
            for v in points:
                classes.setdefault(v % p, []).append(v)
            if len(classes) == p:
                smallest = min(classes.values(), key=len)
                points = [v for v in points if v not in smallest]
                changed = True
                break
    return points


def _local_improve(points, removed):
    """Try to re-add removed elements and perform 1-for-1 swaps to grow size."""
    best = list(points)
    best_set = set(best)
    removed_set = set(removed)
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        # Re-add pass
        for x in sorted(removed_set):
            if _time_left() < 5.0:
                break
            trial = best + [x]
            if _check_admissible(trial):
                best = trial
                best_set.add(x)
                removed_set.discard(x)
                improved = True
        # Swap pass: remove one kept, add one removed
        if _time_left() < 5.0:
                break
        for x in sorted(removed_set):
            if _time_left() < 5.0:
                break
            for y in best:
                if _time_left() < 5.0:
                    break
                if y == x:
                    continue
                trial = [v for v in best if v != y] + [x]
                if _check_admissible(trial):
                    best = trial
                    best_set.discard(y)
                    best_set.add(x)
                    removed_set.discard(x)
                    removed_set.add(y)
                    improved = True
                    break
    return best


def _greedy_conflict_graph(points):
    """Degree-ordered greedy independent-set on the implicit conflict graph.

    Vertices: integers in `points`.
    Edge between u,v if for some prime p <= len(selected), adding both would
    complete a full residue class mod p. We approximate by: greedily select
    vertices in order of increasing "conflict degree" (number of primes p
    <= N+1 for which u and some already-selected vertex cover all p classes
    is hard; instead we use a proxy: select vertices that keep the set
    admissible, ordering by a heuristic score).

    Practical approach: sort candidates by a static heuristic (e.g., value
    itself, or by how many residue classes they are the sole representative of)
    and greedily add if admissibility holds.
    """
    if not points:
        return []
    selected = []
    selected_set = set()
    # Heuristic: process in increasing order of value (or a score).
    # For a conflict-graph view, low-degree vertices are safer to pick first.
    # Approximate degree: for each prime p <= N+1, count how many other
    # points share the same residue mod p. Lower sum = lower conflict.
    arr = np.asarray(points, dtype=np.int64)
    n = len(points)
    degrees = np.zeros(n, dtype=np.int64)
    for p in _PRIMES_UP_TO_N:
        if p > N + 1:
            break
        residues = arr % p
        # For each vertex, count how many others share its residue mod p.
        # Use bincount per residue to get counts, then map back.
        counts = np.bincount(residues, minlength=p)
        deg_p = counts[residues]
        degrees += deg_p
    # Sort by ascending degree (low-conflict first), break ties by value.
    order = np.argsort(degrees, kind='stable')
    for idx in order:
        if _time_left() < 10.0:
            break
        x = int(arr[idx])
        if x in selected_set:
            continue
        trial = selected + [x]
        if _check_admissible(trial):
            selected.append(x)
            selected_set.add(x)
    return selected


def construct():
    # ---------- Baseline: seed repair on [0, N] ----------
    points = list(range(N + 1))
    points = _greedy_repair(points)
    baseline = points if _check_admissible(points) else []

    # ---------- Conflict-graph greedy (degree-ordered) ----------
    cg_candidate = []
    try:
        if _time_left() > 30.0:
            cg_candidate = _greedy_conflict_graph(list(range(N + 1)))
            if not _check_admissible(cg_candidate):
                cg_candidate = []
    except Exception:
        cg_candidate = []

    # Pick the larger of baseline and conflict-graph greedy
    if len(cg_candidate) > len(baseline):
        best = cg_candidate
    else:
        best = baseline

    removed = [x for x in range(N + 1) if x not in set(best)]

    # ---------- Refinement: local improvement ----------
    try:
        if _time_left() > 10.0:
            improved = _local_improve(best, removed)
            if _check_admissible(improved) and len(improved) > len(best):
                best = improved
    except Exception:
        pass

    # ---------- Final safety: ensure admissible, fall back to baseline ----------
    if not _check_admissible(best):
        fallback = _greedy_repair(list(range(N + 1)))
        best = fallback if _check_admissible(fallback) else []

    return list(best)
