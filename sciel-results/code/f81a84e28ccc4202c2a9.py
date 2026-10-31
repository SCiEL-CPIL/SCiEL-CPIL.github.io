"""Cap set greedy builder: hybrid structural + chaos-breaking priority.

This program combines the "line-blocking potential" structural signal from the
current program with the "chaos-breaking" symmetry-breaking and smooth
quadratic distinctness signals from the top-performing inspirations.

The core insight is that the current program's structural signal (center_dist,
entropy) is too local and symmetric, leading to the greedy algorithm getting
trapped in low-complexity sub-cubes (like {0,1}^n). By adding a strong
symmetry-breaking term (favoring vectors with more 2s) and a smooth quadratic
distinctness score, we force the greedy selection to explore the full {0,1,2}^n
space more uniformly while still respecting the local geometric constraints
imposed by the line-blocking potential.

We also refine the danger proxy by using a smooth variance penalty instead of
the discrete sum-of-squares center distance, and add a fast polynomial hash
for robust tie-breaking to avoid degeneracies in the priority landscape.
"""
import time

_start_time = None
_n_cache = None
_priority_cache = None
_deadline = 320.0  # Safety margin before the 330s kill


def _init(n):
    global _start_time, _n_cache, _priority_cache
    if _start_time is None:
        _start_time = time.time()
    if _n_cache is None or _n_cache != n:
        _n_cache = n
        _priority_cache = {}


def _hybrid_priority(vec, n):
    """
    Deterministic, vector-intrinsic hybrid priority.

    Components:
    1. distinct_score: Smooth quadratic score for using all 3 symbols.
    2. weight_score: Symmetry-breaking bonus for vectors with more 2s.
    3. balance_score: L2 penalty for deviation from uniform symbol distribution.
    4. danger_score: Smooth variance penalty (high variance = more line risk).
    5. blocking_score: Structural line-blocking potential (center + entropy).
    6. hash_score: Fast polynomial hash for deterministic tie-breaking.
    """
    # --- Fast symbol counting in one pass ---
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    for v in vec:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
            s += 1
            s2 += 1
        else:  # v == 2
            c2 += 1
            s += 2
            s2 += 4

    total = n
    if total == 0:
        return 0.0

    # Factor 1: Smooth quadratic distinct-symbol score.
    # Quadratic in presence count: 0 -> 0, 1 -> 10000, 2 -> 40000, 3 -> 90000.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Symmetry breaking via count of 2s.
    # Favors vectors with more 2s to break the symmetry of the ternary grid
    # and prevent the greedy algorithm from clustering in the {0,1}^n sub-cube.
    weight_score = c2 * 100.0

    # Factor 3: Quadratic balance penalty (smooth surrogate for L1).
    # L2 deviation from the uniform (n/3, n/3, n/3) distribution.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Smooth variance-based danger proxy.
    # High coordinate variance = more extreme coordinates = higher line risk.
    mean = s / total
    variance = s2 / total - mean * mean
    danger_score = -variance * 5000.0

    # Factor 5: Structural line-blocking potential (from current program).
    # We retain the entropy and center-distance signals but re-weight them
    # to be secondary to the chaos-breaking terms.
    import numpy as np
    entropy = 0.0
    for cnt in (c0, c1, c2):
        if cnt > 0:
            p = cnt / total
            entropy -= p * np.log(p + 1e-15)
    ent_norm = entropy / np.log(3.0)

    # Center distance: 1 at center (all 1s), 0 at corners (all 0s or 2s).
    # sum_sq = s2. Center is at sum_sq = 2*n.
    max_sum_sq = 4 * n
    center_dist = 1.0 - abs(s2 - 2 * n) / max_sum_sq

    blocking_score = 1.0 * center_dist + 0.7 * ent_norm

    # Factor 6: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vec):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Factor 7: Algebraic degree score in F₃.
    # Treat the vector as coefficients of a polynomial p(x) = sum(v_i * x^i) in F₃[x].
    # The degree is the highest i where v_i != 0. Higher degree means more
    # structured non-linearity, which resists the linear dependencies that
    # define affine lines in F₃ⁿ.
    # We also compute a "degree-weighted energy" that rewards high-degree terms
    # with larger weights, creating a smooth gradient toward complex polynomials.
    # This is a smooth surrogate for the discrete degree: instead of a hard
    # jump at the highest non-zero coefficient, we use exponential decay.
    
    # Compute degree-weighted energy: sum(v_i * exp(-i/n)) for i where v_i != 0
    # This gives more weight to higher-degree terms while remaining smooth.
    import math
    degree_energy = 0.0
    for i, v in enumerate(vec):
        if v != 0:
            # Exponential decay from high degree to low degree
            # At i=0 (constant term): weight = 1.0
            # At i=n-1 (highest degree): weight = exp(-(n-1)/n) ≈ 1/e
            # This creates a smooth preference for high-degree terms
            weight = math.exp(-i / max(n, 1))
            degree_energy += v * weight
    
    # Also compute the actual degree (highest non-zero index) for a discrete bonus
    degree = -1
    for i in range(n - 1, -1, -1):
        if vec[i] != 0:
            degree = i
            break
    
    # Discrete degree bonus: rewards vectors with high degree
    # Scale: degree ranges from -1 (zero vector) to n-1
    degree_bonus = max(0, degree) * 2.0
    
    # Combine: smooth energy + discrete bonus
    mult_score = degree_energy * 10.0 + degree_bonus

    # --- Combine ---
    # The distinct_score and weight_score are the dominant "chaos-breaking" signals.
    # The danger_score and balance_score are the smooth "safety" signals.
    # The blocking_score is the structural "line-blocking" signal.
    # The hash_score is a tiny tie-breaker.
    
    # We scale the blocking_score down slightly so that the global symmetry-breaking
    # terms dominate the local structural terms, preventing the greedy algorithm
    # from getting stuck in local optima defined by the lattice symmetries.
    total_priority = (
        distinct_score
        + weight_score
        + balance_score
        + danger_score
        + 50.0 * blocking_score  # Scale up blocking to be competitive but not dominant
        + mult_score
        + hash_score
    )
    
    # Additional algebraic structure: compute the "algebraic complexity"
    # by checking how many distinct power sums the vector generates.
    # For a vector v, compute S_k = sum(v_i^k) for k=1,2,...,min(n,5).
    # Vectors with more distinct power sums have richer algebraic structure
    # and are less likely to lie on affine lines.
    # This is a smooth, computationally cheap proxy for algebraic independence.
    
    power_sums = set()
    for k in range(1, min(n, 5) + 1):
        s = 0
        for v in vec:
            s += v ** k
        power_sums.add(s)
    
    # Reward having many distinct power sums (up to 5)
    algebraic_score = len(power_sums) * 3.0
    
    total_priority += algebraic_score
    
    return float(total_priority)


def priority(vector, n):
    """
    Returns a priority value for the given vector. Higher priority means the
    vector is considered first in the greedy insertion.

    Strategy: Hybrid of chaos-breaking (symmetry breaking + distinctness)
    and structural line-blocking potential.
    """
    _init(n)

    # Deadline guard: return a fast deterministic fallback near the limit.
    elapsed = time.time() - _start_time
    if elapsed > _deadline:
        vec = tuple(vector)
        # Fast fallback: hash + algebraic degree proxy
        h = 0
        degree = -1
        for i in range(len(vec) - 1, -1, -1):
            if vec[i] != 0:
                degree = i
                break
        for i, val in enumerate(vec):
            h = (h * 31 + val + i) & 0x7FFFFFFF
        return float(h + max(0, degree) * 50.0)

    vec = tuple(vector)

    if vec in _priority_cache:
        return _priority_cache[vec]

    val = _hybrid_priority(vec, n)
    _priority_cache[vec] = val
    return val
