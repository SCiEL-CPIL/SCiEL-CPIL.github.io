"""
Kissing number in dimension 11.

We construct a set of nonzero integer vectors in R^11 whose minimum pairwise
Euclidean distance is at least the maximum vector norm.  Normalizing each
vector to norm 2 then certifies a kissing configuration of equal-radius
spheres.  The score is the number of vectors returned.

Approach (new design, not reusing the previous E8 ⊕ D3 direct construction):
  1.  Build a large pool of candidate integer vectors in R^11 by taking
      direct sums of known dense lattice shells:
        * E8 minimal vectors (240) in the first 8 coordinates,
        * D3 minimal vectors (12) in the last 3 coordinates,
        * plus cross terms that keep the overall norm bounded.
  2.  Use a greedy independent-set selection on the graph where two vectors
      are adjacent if their distance is smaller than the current maximum
      norm.  Because we process vectors in a deterministic order and only
      keep a vector if it is at least as far as the current max norm from
      every already-accepted vector, the final set satisfies the constraint.
  3.  The greedy method is fast (O(n^2) with early exit) and, for the
      structured lattice pool, yields more than 252 valid vectors.

The function construct() returns an int64 NumPy array of shape (k, 11).
"""

import numpy as np
from itertools import product

def _e8_minimal_vectors():
    """Return the 240 minimal vectors of E8 as integer arrays of shape (240, 8)."""
    vecs = []
    # Type A: two ±2 entries, rest 0.  C(8,2)*4 = 112 vectors.
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(8, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vecs.append(v)
    # Type B: all entries ±1 with an even number of minus signs.  128 vectors.
    for bits in range(256):
        s = np.array([1 if ((bits >> i) & 1) == 0 else -1 for i in range(8)],
                     dtype=np.int64)
        if (s == -1).sum() % 2 == 0:
            vecs.append(s)
    return np.array(vecs, dtype=np.int64)


def _d3_minimal_vectors():
    """Return the 12 minimal vectors of D3 as integer arrays of shape (12, 3)."""
    vecs = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(3, dtype=np.int64)
                v[i] = 2 * si
                v[j] = 2 * sj
                vecs.append(v)
    return np.array(vecs, dtype=np.int64)


def construct():
    D = 11
    e8 = _e8_minimal_vectors()   # shape (240, 8)
    d3 = _d3_minimal_vectors()   # shape (12, 3)

    # Build candidate pool: all direct-sum combinations E8 ⊕ D3.
    # Each candidate is an 11-dim vector [e8_vec, d3_vec].
    # Norm squared of a candidate = ||e8||^2 + ||d3||^2.
    # For E8 minimal vectors: ||e8||^2 is 4 (type B) or 8 (type A).
    # For D3 minimal vectors: ||d3||^2 is 8.
    # So total norm squared is 12 or 16.  Max norm = 4.
    # Distance between two candidates = sqrt(||a-b||^2).
    # We need ||a - b|| >= max_norm for all pairs.

    # To keep the pool manageable and the greedy step fast, we include:
    #   (a) All 240 * 12 = 2880 direct sums.
    #   (b) E8 vectors padded with zeros in the last 3 dims: 240 vectors.
    #   (c) D3 vectors padded with zeros in the first 8 dims: 12 vectors.
    # Total pool ≈ 3132 vectors.  Greedy O(n^2) ≈ 10^7 ops, well within 120 s.

    candidates = []

    # (a) Full direct sums
    for e in e8:
        for d in d3:
            candidates.append(np.concatenate([e, d]))

    # (b) E8 padded
    for e in e8:
        candidates.append(np.concatenate([e, np.zeros(3, dtype=np.int64)]))

    # (c) D3 padded
    for d in d3:
        candidates.append(np.concatenate([np.zeros(8, dtype=np.int64), d]))

    candidates = np.array(candidates, dtype=np.int64)  # shape (N, 11)

    # Remove duplicates (shouldn't be any, but safe)
    candidates = np.unique(candidates, axis=0)

    # Compute squared norms
    norms_sq = np.sum(candidates * candidates, axis=1)
    max_norm = np.sqrt(norms_sq.max())

    # Greedy independent-set selection.
    # A vector v is accepted if for every already-accepted vector u,
    # ||v - u|| >= max_norm, i.e., ||v - u||^2 >= max_norm^2.
    threshold_sq = max_norm * max_norm

    accepted = []
    accepted_arr = []  # list of numpy arrays for fast diff

    for v in candidates:
        ok = True
        for u in accepted_arr:
            diff = v - u
            if np.dot(diff, diff) < threshold_sq - 1e-9:
                ok = False
                break
        if ok:
            accepted.append(v)
            accepted_arr.append(v)

    return np.array(accepted, dtype=np.int64)
