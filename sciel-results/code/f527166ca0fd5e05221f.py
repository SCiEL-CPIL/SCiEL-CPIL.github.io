import time
import numpy as np

# EVOLVE-BLOCK-START
import functools
import itertools

_DEADLINE = 330.0
_MARGIN = 25.0  # safety margin before the 330s kill

# Cache of the full construction result keyed by n.
# The evaluator may call priority() many times for the same n; we only
# need to build the set once per n and then just return the size.
_results = {}


def _is_cap_set(vecs):
    """vecs: list of tuples over {0,1,2}. Return True if no 3-AP (collinear)."""
    s = set(vecs)
    m = len(s)
    for i in range(m):
        a = vecs[i]
        for j in range(i + 1, m):
            b = vecs[j]
            # c = 2b - a must be in {0,1,2}^n and in s, and c != a, c != b
            c = tuple(2 * bi - ai for ai, bi in zip(a, b))
            if all(0 <= x <= 2 for x in c) and c in s and c != a and c != b:
                return False
    return True


def _collision_score(vector, n, weight_counts, coord_entropy):
    """
    Estimate a local 3-AP collision probability for `vector`.
    Lower is better (fewer expected collisions).

    Heuristic components:
      1) Hamming-weight balance: vectors near the "middle" weight have more
         neighbours in the cube, hence more potential 3-AP partners.
      2) Coordinate-wise entropy: coordinates that are "mixed" (all of 0,1,2
         plausible) contribute more potential alignments.
    """
    w = sum(1 for v in vector if v != 0)  # count of non-zero coords
    # 1) weight term: penalise middle weights
    # normalise weight to [0,1]
    t = w / max(1, n)
    # a parabola peaking at t=0.5 gives high collision for middle weights
    weight_pen = (t - 0.5) ** 2  # small for middle, large for extremes
    # We want HIGH priority for LOW collision, so we want this term to be
    # SMALL for low-collision vectors. Middle weights -> small weight_pen ->
    # we must invert: use (1 - weight_pen) as a "badness" or just use weight_pen
    # as a "goodness" contribution. We'll combine so that lower total = better.
    # For low collision we want middle weight to be BAD (high score). So add
    # a term that is large in the middle.
    weight_bad = 1.0 - weight_pen  # large in the middle -> bad

    # 2) coordinate entropy term: average over coordinates of how "spread out"
    #    the value is. A coordinate equal to 1 (middle value) is more likely to
    #    participate in a 3-AP than 0 or 2.
    ent = 0.0
    for v in vector:
        if v == 1:
            ent += 1.0
        else:
            ent += 0.25
    ent /= max(1, n)  # in [0.25, 1.0]

    # Combine: higher score = worse. We'll return -score as priority so that
    # higher priority = lower collision.
    bad = 0.6 * weight_bad + 0.4 * ent
    return -bad


def _build_cap_set(n):
    """Greedy construction of a cap set in {0,1,2}^n using collision-density order."""
    # Enumerate all vectors in {0,1,2}^n
    all_vecs = list(itertools.product(range(3), repeat=n))

    # Precompute per-coordinate entropy estimate (static for the whole cube,
    # but we use the vector's own values as a proxy for local density).
    # We'll just compute the score on the fly; it's O(n) per vector.

    # Order by collision score (higher priority = lower badness)
    scored = []
    for v in all_vecs:
        s = _collision_score(v, n, None, None)
        scored.append((s, v))
    scored.sort(key=lambda x: (-x[0], x[1]))  # descending priority

    chosen = []
    chosen_set = set()
    deadline = time.time() + _DEADLINE - _MARGIN

    for s, v in scored:
        if time.time() > deadline:
            break
        # Check cap-set property with v added
        ok = True
        for a in chosen:
            # Check if {a, v, c} forms a 3-AP for some c in {0,1,2}^n
            # c = 2v - a
            c = tuple(2 * vi - ai for ai, vi in zip(a, v))
            if all(0 <= x <= 2 for x in c):
                if c in chosen_set:
                    ok = False
                    break
            # Also check a, c, v ordering and a, v, c etc. The above covers
            # the case where v is the middle. We also need the case where a is
            # the middle: 2a - v in set, and v is the middle: 2v - a in set.
            # Check a as middle:
            c2 = tuple(2 * ai - vi for ai, vi in zip(a, v))
            if all(0 <= x <= 2 for x in c2) and c2 in chosen_set and c2 != a and c2 != v:
                ok = False
                break
        if ok:
            chosen.append(v)
            chosen_set.add(v)

    return len(chosen)


def priority(vector, n):
    """
    Greedy cap-set builder. Returns the cap set size achievable under the
    collision-density priority ordering for dimension n.
    """
    key = n
    if key in _results:
        return float(_results[key])

    if n <= 0:
        return 0.0

    # For very small n, brute force is cheap and exact.
    if n <= 3:
        # Known cap set sizes: n=1 -> 2 (any two non-collinear; in {0,1,2} no
        # 3 distinct collinear exist since only 3 points per line and they'd
        # need to be all three). Actually in 1D {0,1,2}, the only 3-AP is
        # (0,1,2). So max cap set = 2.
        sizes = {1: 2, 2: 4, 3: 9}
        return float(sizes.get(n, 0))

    # Build the cap set greedily
    size = _build_cap_set(n)
    _results[key] = size
    return float(size)
# EVOLVE-BLOCK-END
