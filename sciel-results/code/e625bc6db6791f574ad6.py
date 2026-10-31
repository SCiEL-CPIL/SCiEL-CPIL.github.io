# EVOLVE-BLOCK-START
import time
import math

_start_time = None
_deadline = 315.0  # safety margin before the 330s kill

def priority(vector, n):
    """
    Priority for greedy cap-set construction in {0,1,2}^n.

    Structural Strategy:
    1. Soft-Quadratic Distinctness: Smooth preference for vectors using all 3 symbols
       via log-sum-exp style soft-max, creating a continuous gradient.
    2. L2 Balance Penalty: Penalize deviation from uniform symbol distribution
       using squared error (smooth surrogate for L1).
    3. Variance Penalty: Penalize high coordinate variance (extreme coords form APs).
    4. Fourier Sparsity Proxy: Penalize coordinates with non-uniform value distribution
       (high local spectral energy correlates with AP completion risk).
    5. Pairwise Correlation Penalty: Penalize coordinate pairs that frequently align
       in AP-like patterns (smooth surrogate for global AP density).
    6. Symmetry Breaking: Favor vectors with more 2s to prevent {0,1}^n clustering.
    7. Deterministic Hash: Fast polynomial hash for tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _deadline:
        return 0.0

    # Fast symbol counting in one pass
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    # Per-coordinate value counts for Fourier proxy
    # For n up to ~10, we can track per-coordinate distributions
    if n <= 12:
        coord_counts = [[0, 0, 0] for _ in range(n)]
    else:
        coord_counts = None

    for i, v in enumerate(vector):
        if v == 0:
            c0 += 1
            if coord_counts is not None:
                coord_counts[i][0] += 1
        elif v == 1:
            c1 += 1
            s += 1
            s2 += 1
            if coord_counts is not None:
                coord_counts[i][1] += 1
        else:  # v == 2
            c2 += 1
            s += 2
            s2 += 4
            if coord_counts is not None:
                coord_counts[i][2] += 1

    # Factor 1: Soft-quadratic distinctness via smooth indicator.
    # Use soft presence: p_k = 1/(1+exp(-alpha * count_k)) for each symbol k.
    # Then distinctness = sum of pairwise products (smooth count of symbol pairs present).
    alpha = 3.0
    p0 = 1.0 / (1.0 + math.exp(-alpha * (c0 / max(n, 1))))
    p1 = 1.0 / (1.0 + math.exp(-alpha * (c1 / max(n, 1))))
    p2 = 1.0 / (1.0 + math.exp(-alpha * (c2 / max(n, 1))))
    # Smooth pairwise presence: counts how many pairs of symbols are "effectively present"
    soft_distinct = p0 * p1 + p1 * p2 + p0 * p2
    distinct_score = 120000.0 * soft_distinct

    # Factor 2: Symmetry breaking via count of 2s.
    weight_score = c2 * 100.0

    # Factor 3: Quadratic balance penalty.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Smooth variance-based danger proxy.
    mean = s / n if n > 0 else 0.0
    variance = s2 / n - mean * mean if n > 0 else 0.0
    danger_score = -variance * 5000.0

    # Factor 5: Fourier sparsity proxy.
    # For each coordinate, measure how non-uniform the value distribution is.
    # A coordinate that is always the same value has high "spectral energy" at DC,
    # which in Z_3^n correlates with AP risk. We use chi-squared statistic per coord.
    fourier_penalty = 0.0
    if coord_counts is not None and n > 0:
        for cc in coord_counts:
            # Chi-squared: sum((obs - exp)^2 / exp) where exp = n/3 for each symbol
            exp_val = n / 3.0
            if exp_val > 0:
                chi2 = sum((obs - exp_val) ** 2 / exp_val for obs in cc)
                fourier_penalty += chi2
        # Normalize by number of coordinates for scale consistency
        fourier_penalty /= n
        fourier_score = -fourier_penalty * 800.0
    else:
        # For large n, use a global proxy: the total chi-squared of overall distribution
        exp_val = n / 3.0
        if exp_val > 0:
            chi2_global = (c0 - exp_val) ** 2 / exp_val + (c1 - exp_val) ** 2 / exp_val + (c2 - exp_val) ** 2 / exp_val
            fourier_score = -chi2_global * 50.0
        else:
            fourier_score = 0.0

    # Factor 6: Pairwise correlation penalty (AP alignment proxy).
    # For pairs of coordinates (i,j), count how often they form an AP-aligned pattern.
    # In Z_3, a pair (a,b) is "aligned" if b - a is constant across many positions.
    # Efficient proxy: count how many coordinate pairs have the same value.
    # High same-value correlation suggests structure that can extend to APs.
    # Use a cheap surrogate: count of (v_i == v_j) for sampled pairs.
    if n <= 10:
        same_count = 0
        total_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1
                if vector[i] == vector[j]:
                    same_count += 1
        # Expected same_count under random: n*(n-1)/2 * (1/3)
        expected_same = total_pairs / 3.0
        same_dev = same_count - expected_same
        correlation_score = -same_dev * 200.0
    elif n <= 20:
        # Sample pairs for larger n
        import random
        # Use deterministic sampling based on vector hash for reproducibility
        seed_val = hash(vector) & 0xFFFFFFFF
        rng = random.Random(seed_val)
        sample_size = min(200, n * (n - 1) // 2)
        same_count = 0
        for _ in range(sample_size):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i != j:
                if vector[i] == vector[j]:
                    same_count += 1
        expected_same = sample_size / 3.0
        same_dev = same_count - expected_same
        correlation_score = -same_dev * 200.0
    else:
        correlation_score = 0.0

    # Factor 7: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first.
    total = distinct_score + weight_score + balance_score + danger_score + fourier_score + correlation_score + hash_score
    return float(total)
# EVOLVE-BLOCK-END
