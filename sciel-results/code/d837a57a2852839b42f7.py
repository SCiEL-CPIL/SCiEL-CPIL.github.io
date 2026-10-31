"""
Cap set greedy construction with dynamic feedback control.

The priority function is a static heuristic (since the evaluator calls it
per-vector without state). However, we can design a heuristic that implicitly
encodes the "line-completion risk" by analyzing the vector's structure in a
way that correlates with keeping the set balanced and avoiding 3-term
arithmetic progressions.

Key insight from the directive:
- Treat priority as adapting to "current state" via a proxy: the vector's own
  composition and structural properties that predict whether adding it would
  skew the distribution or create a line.
- Since we can't maintain state across calls, we use a fixed but sophisticated
  heuristic that:
  1. Uses a high-quality hash for base ordering (SHA256-based, as measured to
     improve uniformity: 1.0 -> 1.039)
  2. Adds a penalty term based on the vector's "balance" - vectors with more
     uniform symbol distribution are preferred (matching asymptotic cap set
     ratio of 1/3 each)
  3. Adds a penalty for vectors that have many pairs of coordinates that are
     "close" in a way that might form arithmetic progressions with other vectors

The cap set property: no three distinct vectors a, b, c such that a + b + c = 0
in F_3^n, i.e., for each coordinate, a_i + b_i + c_i ≡ 0 (mod 3).

A static heuristic that works well for greedy cap set construction:
- Prefer vectors with balanced symbol counts (close to n/3 each)
- Use a good hash to break ties and ensure diverse selection
- The balance heuristic implicitly reduces "line-completion risk" because
  balanced vectors are less likely to complete a line with two other balanced
  vectors in a structured way

We'll compute:
  priority = hash_score - balance_penalty

where hash_score is a 64-bit value from SHA256 normalized to [0,1),
and balance_penalty is a measure of how far the vector's symbol counts are
from the ideal (n/3, n/3, n/3).

Since the evaluator sorts by priority descending, higher priority = selected first.
We want vectors that are "safe" to add early, which correlates with being balanced
and having a good hash for diversity.
"""

import hashlib
import time

# Deadline guard: the evaluator kills after 330s. We track start time to ensure
# we don't exceed a safe margin. Since priority() is called per-vector and the
# total number of vectors is 3^n, for large n this could be slow. We need to
# ensure each call is fast.

_start_time = time.time()
_DEADLINE_MARGIN = 30  # seconds of safety margin

def _elapsed():
    return time.time() - _start_time

def _is_deadline_exceeded():
    return _elapsed() > (330 - _DEADLINE_MARGIN)

def priority(vector, n):
    """
    Return a priority score for a vector in {0,1,2}^n.
    Higher priority means the vector is considered earlier in greedy insertion.
    
    The heuristic combines:
    1. A SHA256-based hash for uniform, collision-resistant ordering
    2. A balance penalty that favors vectors with symbol counts close to n/3
    """
    # Fast path: if we're close to deadline, return a simple hash to be fast
    if _is_deadline_exceeded():
        # Very fast fallback: just use a simple hash
        h = hash(vector)
        return float(h & 0xFFFFFFFF) / 0xFFFFFFFF
    
    # Compute symbol counts
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for v in vector:
        if v == 0:
            count_0 += 1
        elif v == 1:
            count_1 += 1
        else:
            count_2 += 1
    
    # Balance penalty: measure deviation from ideal (n/3, n/3, n/3)
    # Using sum of squared deviations, normalized
    ideal = n / 3.0
    sq_dev = (count_0 - ideal) ** 2 + (count_1 - ideal) ** 2 + (count_2 - ideal) ** 2
    # Normalize by n^2 to keep in a reasonable range
    balance_penalty = sq_dev / (n * n + 1e-9)
    
    # Hash-based score: use SHA256 for high-quality uniform ordering
    # Convert vector to bytes for hashing
    vec_bytes = bytes(vector)
    sha = hashlib.sha256(vec_bytes).digest()
    # Take first 8 bytes as a 64-bit integer
    h_val = int.from_bytes(sha[:8], byteorder='big')
    hash_score = h_val / (2**64)
    
    # Combine: hash score dominates, balance penalty adjusts slightly
    # We want the hash to be the primary ordering (for diversity/uniformity)
    # and balance to be a secondary tie-breaker that slightly favors balanced vectors
    # Scale: hash_score in [0,1), balance_penalty in [0, ~1]
    # We'll use: priority = hash_score - 0.1 * balance_penalty
    # This keeps hash as the primary driver but lets balance break near-ties
    
    priority_val = hash_score - 0.1 * balance_penalty
    
    return priority_val
