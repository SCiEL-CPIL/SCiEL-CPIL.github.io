import time

_START_TIME = time.time()
_TIME_LIMIT = 270.0  # Safety margin before 300s hard limit


def priority(vector, n):
    # Deadline guard: if we've overshot, return a neutral low score
    # so the evaluator can still finish its sort/insertion pass.
    if time.time() - _START_TIME > _TIME_LIMIT:
        return 0.0

    # Count symbols 0, 1, 2
    n0 = 0
    n1 = 0
    n2 = 0
    for v in vector:
        if v == 0:
            n0 += 1
        elif v == 1:
            n1 += 1
        else:
            n2 += 1

    # ---- Spectral / Fourier analysis on Z_3^n ----
    # A 3-term AP a,b,c in Z_3^n satisfies a + c = 2b (mod 3).
    # The Fourier transform of a single vector v decomposes into n independent
    # coordinate factors.  The magnitude of the Fourier coefficient at the
    # trivial character (all-ones) is |v| = 1 (it's a point mass), but what
    # matters for AP-density is how the vector's symbol distribution aligns
    # with low-frequency characters.
    #
    # Key insight: for a vector v in {0,1,2}^n, the number of 3-APs it can
    # participate in as a CENTER is proportional to the product over coordinates
    # of the "midpoint affinity".  A coordinate value of 1 is the midpoint of
    # (0,2) and (2,0), so it has high midpoint affinity.  Values 0 and 2 are
    # midpoints of (2,1) and (1,2) respectively -- also midpoint-capable but
    # in a rotated frame.
    #
    # We compute a proxy for the L2 norm of the Fourier transform restricted
    # to low-frequency characters (those with few non-trivial coordinates).
    # This captures global structural alignment that local counts miss.

    # 1) Midpoint risk: n1 is the count of coordinates equal to 1.
    #    A vector with many 1s is a natural midpoint of pairs (0,2) in those
    #    coordinates, creating many 3-APs.
    midpoint_risk = (n1 * n1) / max(n, 1)

    # 2) Fourier low-frequency energy: estimate the squared magnitude of the
    #    Fourier transform at the trivial and first-order characters.
    #    For a point mass at v, the Fourier coefficient at character chi_k is
    #    prod_i omega^{k_i * v_i} where omega = exp(2*pi*i/3).
    #    The |F(v, chi_k)|^2 = 1 for all k (Plancherel), but the *phase*
    #    determines constructive/destructive interference when summing over
    #    a set.  For a single vector, we use the symbol-count imbalance as
    #    a proxy for how much the vector "projects onto" structured directions.
    #
    #    The variance of symbol counts measures deviation from uniform:
    #    high variance = concentrated on few symbols = aligned with low-freq.
    target = n / 3.0
    balance_var = (n0 - target) ** 2 + (n1 - target) ** 2 + (n2 - target) ** 2

    # 3) 0/2 asymmetry: in a 3-AP over Z3, the pair (0,2) is symmetric around 1.
    #    If n0 and n2 are very different, the vector is "polarised" toward one
    #    endpoint, which also correlates with structured APs.
    zero_two_imbalance = (n0 - n2) ** 2 / max(n, 1)

    # 4) Coordinate-pair interaction term: capture quadratic AP structure.
    #    For a 3-AP in Z_3^n, the constraint a+c=2b is quadratic in the
    #    coordinate values.  The number of pairs of coordinates that can
    #    jointly form a "2D AP" is related to how the symbol counts interact
    #    across coordinate pairs.  We estimate this via the product of
    #    pairwise co-occurrence patterns.
    #    Specifically, count how many coordinate pairs (i,j) have the same
    #    symbol value -- this measures "block structure" that favors APs.
    #    A vector with long runs of identical symbols has high block structure.
    #    We approximate this by the sum over symbols of C(n_s, 2) / C(n, 2),
    #    which is the probability that two random coordinates share a symbol.
    pair_same = (n0 * (n0 - 1) + n1 * (n1 - 1) + n2 * (n2 - 1)) / max(n * (n - 1), 1)

    # 5) Coordinate-level spread: count how many distinct "types" of
    #    coordinate patterns exist.  A vector that is mostly one symbol
    #    (e.g. all 0s) is extremely structured.  We penalise concentration.
    #    Use the largest symbol count as a concentration measure.
    max_count = max(n0, n1, n2)
    concentration = max_count / max(n, 1)  # in [1/3, 1]

    # ---- Combine into a single score (higher = picked earlier) ----
    # We want to AVOID: high midpoint_risk, high balance_var, high
    # zero_two_imbalance, high pair_same, high concentration.
    # So the priority is the negative of a weighted risk.

    risk = (
        3.0 * midpoint_risk
        + 1.5 * (balance_var / max(n, 1))
        + 0.8 * (zero_two_imbalance / max(n, 1))
        + 2.5 * pair_same
        + 2.0 * concentration
    )

    # ---- Deterministic tie-breaker via FNV-1a hash ----
    h = 0x811c9dc5
    for v in vector:
        h ^= v
        h = (h * 0x01000193) & 0xFFFFFFFF
    hash_score = (h % 1000003) / 1000003.0  # small perturbation

    # Return: higher priority for lower-risk vectors.
    # Scale the risk term so the hash only breaks ties.
    return -risk * 1000.0 + hash_score
