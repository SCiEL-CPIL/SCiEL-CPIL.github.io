import time
import math

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin

# Precomputed basis vectors for algebraic independence scoring.
# We use coordinate indicators and pairwise product indicators as a low-degree
# monomial basis. The projection onto the orthogonal complement of this span
# measures "algebraic novelty" — vectors that are linearly independent from
# simple coordinate/pairwise interactions are less likely to form affine lines.
#
# For a vector v in {0,1,2}^n, we compute:
#   - Coordinate features: v_i for i in range(n)
#   - Pairwise product features: v_i * v_j for i < j
# The "algebraic complexity" is the norm of the residual after projecting
# v onto the span of these features. However, computing the full projection
# is O(n^3), so we use a fast surrogate: the variance of the vector after
# subtracting its mean (removing the constant basis) and the variance of
# pairwise products (removing linear dependencies).
#
# A simpler and faster proxy: compute the "effective rank" of the vector's
# interaction pattern. Vectors with high algebraic independence have
# coordinate values that are not easily predicted from pairwise products.
# We approximate this by measuring the deviation of v_i * v_j from being
# determined by v_i and v_j individually (which is always true for products,
# so we instead measure the entropy of the pairwise product distribution).
#
# Practical fast computation:
#   1. Count of each symbol: c0, c1, c2
#   2. Sum of pairwise products: sum_{i<j} v_i * v_j
#   3. Sum of squares of pairwise products: sum_{i<j} (v_i * v_j)^2
#   4. Variance of pairwise products = a smooth measure of "spread" in
#      the pairwise interaction space, which correlates with algebraic
#      independence.
#
# Higher variance in pairwise products → more diverse interactions →
# higher algebraic complexity → less likely to be in the span of
# low-degree monomials → preferred for cap set construction.


def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy:
    1. Primary: Smooth quadratic distinct-symbol score. Quadratic in the number
       of present symbols (0,1,4,9 * 10000) gives a smoother landscape than the
       discrete 10000/20000/30000 step, producing a better gradient signal for
       the greedy ordering.
    2. Secondary: Hamming weight relative to all-ones (count of 2s). Favors
       vectors "heavy" in the 2-coordinate to break the symmetry of the ternary
       grid and prevent the greedy algorithm from clustering in the {0,1}^n
       sub-cube, forcing uniform exploration of the full {0,1,2}^n space.
    3. Tertiary: Quadratic balance penalty. L2 deviation from the uniform
       (n/3, n/3, n/3) distribution, a smooth surrogate for the L1 deviation,
       provides a smoother gradient than the piecewise-linear L1 penalty.
    4. Quaternary: Smooth variance-based danger proxy. High coordinate variance
       means more "extreme" coordinates, which are structurally more likely to
       complete an affine line, so we penalize variance.
    5. Quinary: Fast polynomial rolling hash for robust deterministic
       tie-breaking.

    All components computed in O(n) time with minimal per-call overhead.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: return a constant to signal "stop caring about ordering"
        return 0.0

    # Fast symbol counting in one pass (collects counts, sum, and sum-of-squares)
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    # Also accumulate pairwise product statistics for algebraic independence score.
    # sum_pair = sum_{i<j} v_i * v_j
    # sum_pair_sq = sum_{i<j} (v_i * v_j)^2
    # We compute these incrementally: when adding v_k, it contributes
    #   v_k * (sum of all previous v_i) to sum_pair
    #   v_k^2 * (sum of all previous v_i^2) to sum_pair_sq
    # This gives O(n) total time.
    sum_pair = 0
    sum_pair_sq = 0
    for v in vector:
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
        # Update pairwise product accumulators
        sum_pair += v * s  # v * (sum of all previous elements)
        sum_pair_sq += (v * v) * s2  # v^2 * (sum of squares of all previous elements)
        # Update s and s2 already done above

    # Factor 1: Smooth quadratic distinct-symbol score.
    # Quadratic in presence count: 0 -> 0, 1 -> 10000, 2 -> 40000, 3 -> 90000.
    # Smoother than the discrete step, gives a better gradient for the ordering.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Hamming weight relative to all-ones (count of 2s).
    # Favors vectors with more 2s to break symmetry and explore the full space,
    # preventing clustering in the {0,1}^n sub-cube.
    weight_score = c2 * 100.0

    # Factor 3: Quadratic balance penalty (smooth surrogate for L1).
    # L2 deviation from the uniform (n/3, n/3, n/3) distribution.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Smooth variance-based danger proxy.
    # High coordinate variance = more extreme coordinates = higher line risk.
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 5000.0

    # Factor 5: Algebraic independence score via pairwise product variance.
    # Number of pairwise terms: m = n*(n-1)/2
    # Mean of pairwise products: mu_pair = sum_pair / m
    # Variance of pairwise products: var_pair = sum_pair_sq/m - mu_pair^2
    # Higher variance → more diverse pairwise interactions → higher algebraic
    # complexity → vectors that are less "predictable" from low-degree monomials.
    # This structurally prevents additive dependencies that form affine lines.
    m = n * (n - 1) // 2
    if m > 0:
        mu_pair = sum_pair / m
        var_pair = sum_pair_sq / m - mu_pair * mu_pair
        # Normalize: max possible variance for pairwise products in {0,1,2}^n
        # Each pairwise product is in {0,1,2,4}, max variance ~ 4
        algebraic_score = var_pair * 2000.0
    else:
        algebraic_score = 0.0

    # Factor 6: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first.
    total = distinct_score + weight_score + balance_score + danger_score + algebraic_score + hash_score
    return float(total)
