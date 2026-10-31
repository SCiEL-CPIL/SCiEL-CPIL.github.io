import math
import time
import zlib
from collections import Counter

# Global deadline tracking
_start_time = None
_DEADLINE_FRACTION = 0.92  # Use 92% of time budget for safety

# Precomputed basis vectors for algebraic independence scoring
# We use low-degree monomials: coordinate indicators and pairwise products
# These form the "span" we want vectors to be orthogonal to

def _compute_algebraic_score(vector, n):
    """
    Compute a score based on how well the vector projects onto the
    orthogonal complement of low-degree monomial basis vectors.
    
    The idea: cap sets avoid affine lines, which correspond to additive
    dependencies. By measuring projection onto the orthogonal complement
    of simple monomial features, we encourage algebraic independence.
    
    We use:
    - Linear features: indicator of each coordinate being 0, 1, or 2
    - Quadratic features: products of pairs of coordinate indicators
    
    The score is the norm of the residual after projecting out these
    low-degree components, which measures "higher-order" structure.
    """
    if n < 2:
        return 0.0
    
    # Convert vector to centered values for better numerical behavior
    # Map {0,1,2} to {-1, 0, 1} for symmetry
    centered = [v - 1 for v in vector]
    
    # Compute linear feature projections
    # For each coordinate i, we have three basis vectors:
    # e_i^0, e_i^1, e_i^2 (indicator that coord i equals 0, 1, 2)
    
    # Instead of full projection, compute variance of centered values
    # as a proxy for linear independence from constant vectors
    
    mean_val = sum(centered) / n
    linear_residual_sq = 0.0
    for v in centered:
        diff = v - mean_val
        linear_residual_sq += diff * diff
    
    # Compute pairwise interaction features
    # For pairs (i,j), compute centered product c_i * c_j
    # The "projection" removes these, leaving higher-order terms
    
    # Approximate quadratic residual: total variance minus explained variance
    # from linear and pairwise terms
    
    # Compute sum of squared pairwise products
    pair_sum_sq = 0.0
    count_pairs = 0
    for i in range(n):
        ci = centered[i]
        for j in range(i + 1, n):
            cj = centered[j]
            prod = ci * cj
            pair_sum_sq += prod * prod
            count_pairs += 1
    
    if count_pairs > 0:
        mean_pair = pair_sum_sq / count_pairs  # This isn't quite right
        # Actually, let's compute the mean of pairwise products
        pair_prod_sum = 0.0
        for i in range(n):
            ci = centered[i]
            for j in range(i + 1, n):
                pair_prod_sum += centered[i] * centered[j]
        mean_pair_prod = pair_prod_sum / count_pairs if count_pairs > 0 else 0.0
        
        # Variance of pairwise products
        pair_var = 0.0
        for i in range(n):
            ci = centered[i]
            for j in range(i + 1, n):
                prod = ci * centered[j]
                diff = prod - mean_pair_prod
                pair_var += diff * diff
        pair_var /= count_pairs if count_pairs > 0 else 1
    else:
        pair_var = 0.0
    
    # The algebraic independence score is higher when:
    # 1. Linear residual is large (not explained by mean)
    # 2. Pairwise interactions have high variance (not degenerate)
    
    linear_score = linear_residual_sq / (n * 4.0)  # Normalize by max possible
    pair_score = pair_var / 1.0  # Pair products are in {-1,0,1}
    
    return linear_score + 0.5 * pair_score


def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Prioritize vectors with all 3 distinct symbols.
    2. Secondary: Maximize algebraic independence via projection onto
       orthogonal complement of low-degree monomial basis vectors.
    3. Tertiary: Maximize uniformity of symbol distribution.
    4. Quaternary: Maximize Shannon Entropy as a complementary spread measure.
    5. Quinary: Robust hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Check deadline
    if time.time() - _start_time > 300 * _DEADLINE_FRACTION:
        return -1e10

    # Count symbols
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Distinct count (3 is best, 2 is middle, 1 is worst)
    counts = [c0, c1, c2]
    distinct_count = sum(1 for c in counts if c > 0)
    
    # Algebraic independence score (NEW)
    alg_score = _compute_algebraic_score(vector, n)
    
    # Uniformity score: reward being close to n/3 for each symbol
    # Use negative squared deviation from uniform as a smooth surrogate
    uniform_target = n / 3.0
    uniformity_penalty = 0.0
    for c in counts:
        diff = c - uniform_target
        uniformity_penalty += diff * diff
    # Normalize and invert: lower penalty is better
    uniformity_score = -uniformity_penalty / (n * n)
    
    # Shannon Entropy
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            entropy -= p * math.log(p)
    
    # Pairwise diversity: fraction of coordinate pairs with different values
    total_pairs = n * (n - 1) // 2
    if total_pairs > 0:
        same_pairs = (c0 * (c0 - 1) // 2) + (c1 * (c1 - 1) // 2) + (c2 * (c2 - 1) // 2)
        diverse_pairs = total_pairs - same_pairs
        diverse_ratio = diverse_pairs / total_pairs
    else:
        diverse_ratio = 0.0
    
    # Primary score: distinct count
    distinct_score = distinct_count * 10000.0
    
    # Secondary score: Algebraic independence (NEW - high weight)
    alg_scaled = alg_score * 5000.0
    
    # Tertiary score: Higher uniformity (less penalty) is better
    uniformity_scaled = uniformity_score * 2000.0
    
    # Quaternary score: Higher entropy is better
    entropy_score = entropy * 500.0
    
    # Quinary score: Higher pairwise diversity is better
    diverse_score = diverse_ratio * 300.0
    
    # Quinary: Robust hash
    h = 0
    base = 31
    for i, val in enumerate(vector):
        h = (h * base + val + i * 13) % (10**9 + 7)
    
    hash_score = h / (10**9 + 7)
    
    return float(distinct_score + alg_scaled + uniformity_scaled + entropy_score + diverse_score + hash_score)
