"""Fresh design: greedy independent set on small-norm integer vectors in R^11.

The constraint is: for all pairs u, v in S, ||u - v||_2 >= max_{w in S} ||w||_2.
We restrict to vectors with squared norm <= 3 (norm <= sqrt(3)).
Then the max norm squared is at most 3, and we need ||u-v||^2 >= 3 for all pairs.
For u, v with norm^2 <= 3: ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v>.
So we need ||u||^2 + ||v||^2 - 2<u,v> >= 3, i.e. <u,v> <= (||u||^2 + ||v||^2 - 3)/2.

We generate all integer vectors with norm^2 in {2, 3} (skip norm^2=1 since
norm-1 vectors: there are 22, and <u,v> for two distinct norm-1 vectors is 0 or +/-1,
and ||u-v||^2 = 1+1-2<u,v> which is 2 or 4 or 0. If <u,v>=1 they're the same vector.
If <u,v>=0, ||u-v||^2=2 < 3, conflict. If <u,v>=-1, ||u-v||^2=4 >= 3, OK.
So we can have at most one pair of opposite norm-1 vectors... but let's focus on
norm^2 >= 2 for higher density.

Candidate set: all integer vectors with L2 norm squared = 2 or 3.
- norm^2=2: choose 2 coordinates from 11, each ±1. Count = C(11,2)*4 = 55*4 = 220.
- norm^2=3: either 3 coords with ±1 (C(11,3)*8 = 165*8 = 1320), or 1 coord with ±1 and 1 coord with ±sqrt(2)... no, integer only. So just 3 coords each ±1: 1320.

Total candidates: 1540.

Conflict: u, v conflict if ||u-v||^2 < 3, i.e. ||u-v||^2 <= 2.
For integer vectors, ||u-v||^2 is a non-negative integer. So conflict iff ||u-v||^2 <= 2.
||u-v||^2 = sum (ui-vi)^2. For this to be <= 2, the difference vector must have
squared norm 1 or 2 (can't be 0 since u != v).
- norm^2=1: differ in exactly one coordinate by ±1.
- norm^2=2: differ in exactly one coordinate by ±2, OR differ in two coordinates by ±1 each.

So two candidate vectors conflict if their difference has L2 norm 1 or 2.

Strategy: greedy independent set with multiple randomized restarts, picking the
vertex with minimum remaining degree (or randomized among low-degree vertices).
"""

import time
import numpy as np
import itertools
import random

D = 11
DEADLINE = 110  # safety margin before 120s limit


def _generate_candidates():
    """Generate all integer vectors in R^11 with squared norm 2 or 3."""
    candidates = []
    # Norm squared = 2: two coordinates are ±1, rest 0
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                candidates.append(tuple(v))
    # Norm squared = 3: three coordinates are ±1, rest 0
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(tuple(v))
    return candidates


def _conflict(u, v):
    """Check if two integer vectors conflict (||u-v||^2 <= 2)."""
    # Fast check using element-wise difference
    diff_sq = 0
    for a, b in zip(u, v):
        d = a - b
        diff_sq += d * d
        if diff_sq > 2:
            return True
    return diff_sq <= 2 and diff_sq >= 1  # diff_sq=0 means same vector


def _build_conflict_table(candidates):
    """Build a bitarray-style conflict table for fast lookups."""
    n = len(candidates)
    # For memory efficiency with 1540 vertices, use a list of sets
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _conflict(candidates[i], candidates[j]):
                conflicts[i].add(j)
                conflicts[j].add(i)
    return conflicts


def _greedy_is(candidates, conflicts, rng, strategy="min_degree"):
    """Greedy independent set with randomized tie-breaking."""
    n = len(candidates)
    available = set(range(n))
    selected = []
    
    while available:
        if strategy == "min_degree":
            # Pick vertex with minimum degree in the induced subgraph on available
            best_v = None
            best_deg = None
            # For speed, sample if available is large
            if len(available) > 200:
                sample = rng.sample(list(available), min(50, len(available)))
            else:
                sample = list(available)
            for v in sample:
                deg = len(conflicts[v] & available)
                if best_deg is None or deg < best_deg:
                    best_deg = deg
                    best_v = v
            # Also try a few random vertices
            for _ in range(min(5, len(available))):
                v = rng.choice(list(available))
                deg = len(conflicts[v] & available)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
        elif strategy == "random":
            best_v = rng.choice(list(available))
        else:
            best_v = min(available)
        
        selected.append(best_v)
        # Remove best_v and all its neighbors from available
        available.discard(best_v)
        for nb in conflicts[best_v]:
            available.discard(nb)
    
    return selected


def _verify_and_count(candidates, selected_indices):
    """Verify the independent set and return count."""
    sel = [candidates[i] for i in selected_indices]
    # Check all pairs
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            diff_sq = sum((a - b) ** 2 for a, b in zip(sel[i], sel[j]))
            if diff_sq < 3:
                return 0  # invalid
    return len(sel)


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    candidates = _generate_candidates()
    n = len(candidates)
    
    # Build conflict table
    conflicts = _build_conflict_table(candidates)
    
    best_count = 0
    best_selected = None
    
    # Multiple restarts with different strategies and seeds
    for restart in range(500):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        strategy = rng.choice(["min_degree", "min_degree", "random"])
        selected = _greedy_is(candidates, conflicts, rng, strategy)
        count = _verify_and_count(candidates, selected)
        if count > best_count:
            best_count = count
            best_selected = selected
    
    if best_selected is None:
        # Fallback: just return the seed vectors
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    result = np.array([candidates[i] for i in best_selected], dtype=np.int64)
    return result


# Pre-generate at module level to save time in construct()
_CANDIDATES = _generate_candidates()
_CONFLICTS = _build_conflict_table(_CANDIDATES)


def construct():
    global _CANDIDATES, _CONFLICTS
    start_time = time.time()
    rng = random.Random(12345)
    
    candidates = _CANDIDATES
    conflicts = _CONFLICTS
    n = len(candidates)
    
    best_count = 0
    best_selected = None
    
    # Phase 1: Deterministic greedy (min degree, no randomization)
    selected = _greedy_is(candidates, conflicts, rng, "min_degree")
    count = _verify_and_count(candidates, selected)
    if count > best_count:
        best_count = count
        best_selected = selected
    
    # Phase 2: Randomized restarts
    restart = 0
    while time.time() - start_time < DEADLINE - 10:
        strategy = rng.choice(["min_degree", "min_degree", "min_degree", "random"])
        selected = _greedy_is(candidates, conflicts, rng, strategy)
        count = _verify_and_count(candidates, selected)
        if count > best_count:
            best_count = count
            best_selected = selected
        restart += 1
    
    if best_selected is None:
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    result = np.array([candidates[i] for i in best_selected], dtype=np.int64)
    return result
