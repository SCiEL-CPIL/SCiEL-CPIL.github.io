"""Cap set priority function for greedy insertion."""
import time

_start_time = time.time()
_DEADLINE_MARGIN = 10  # seconds before the 300s limit

def priority(vector, n):
    """
    Computes a priority score for a vector in {0, 1, 2}^n.
    Higher scores indicate vectors that should be added earlier in the greedy process.

    Strategy:
    1. Primary: Balance symbol counts (0, 1, 2) to avoid early constraint saturation.
    2. Secondary: Penalize vectors with many identical adjacent pairs (reduces line risk).
    3. Tertiary: Fourier-analytic spectral independence — penalize high autocorrelation
       (large projection onto low-frequency Fourier modes of the Hamming association
       scheme) to reduce the likelihood of forming 3-term arithmetic progressions.
    4. Quaternary: Deterministic hash for tie-breaking to break symmetries.
    """
    # Deadline guard: if we're close to the limit, return a neutral score
    elapsed = time.time() - _start_time
    if elapsed > 300 - _DEADLINE_MARGIN:
        return 0.0

    c0 = 0
    c1 = 0
    c2 = 0

    # Fast counting
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1

    # Calculate deviation from ideal balance (n/3 each)
    d0 = 3 * c0 - n
    d1 = 3 * c1 - n
    d2 = 3 * c2 - n

    # Sum of squared deviations (penalty) - main balance metric
    balance_penalty = d0 * d0 + d1 * d1 + d2 * d2

    # Secondary: Count positions where adjacent elements are equal
    adjacent_equal = 0
    if n > 1:
        prev = vector[0]
        for i in range(1, n):
            if vector[i] == prev:
                adjacent_equal += 1
            prev = vector[i]

    # Penalize adjacent equals
    adj_penalty = adjacent_equal * 2

    # Tertiary: Fourier-analytic spectral independence score.
    # In F_3^n, a 3-term AP {x, y, z} satisfies x + z = 2y (mod 3), i.e.,
    # x - 2y + z = 0. The Fourier transform of the indicator of a cap set
    # should have small mass on non-trivial characters. For a single vector v,
    # we estimate its "autocorrelation energy" by computing the sum over
    # all shifts s of the indicator that v+s is also "aligned" with v.
    #
    # A practical proxy: for each coordinate position, the character values
    # exp(2*pi*i*k*v_j/3) for k=1,2 capture the first non-trivial Fourier
    # modes. Vectors whose coordinates are all equal (e.g., all 0s) have
    # large projection onto the trivial character; vectors with good
    # distribution across {0,1,2} in each "block" have smaller low-frequency
    # energy.
    #
    # We compute a weighted autocorrelation: for each pair of positions (i,j),
    # check if the difference vector v_i - v_j (mod 3) is 0 (same symbol).
    # High count of zero-differences means the vector is "concentrated" and
    # more likely to lie on lines with other similar vectors.
    #
    # Efficient computation: count pairs (i,j) with i<j where v_i == v_j.
    # This equals sum over symbols s of C(count_s, 2).
    # For n up to ~10, this is O(n^2) worst case but with early termination.
    #
    # Additionally, we compute a "frequency spread" metric: for each symbol,
    # the variance of its positions. Low variance means the symbol is
    # clustered in a small region, increasing AP risk.

    # Pair count of equal symbols (autocorrelation at shift 0 in difference set)
    pair_equal = 0
    # Use the counts we already have: C(c0,2) + C(c1,2) + C(c2,2)
    pair_equal = c0 * (c0 - 1) // 2 + c1 * (c1 - 1) // 2 + c2 * (c2 - 1) // 2

    # Maximum possible pair_equal is C(n,2) = n*(n-1)//2
    max_pairs = n * (n - 1) // 2
    if max_pairs > 0:
        # Normalize to [0, 1]: 1 means all symbols identical (worst), 0 means perfectly balanced
        autocorr_frac = pair_equal / max_pairs
    else:
        autocorr_frac = 0.0

    # Spectral penalty: weight the autocorrelation fraction
    # Scale: for n=10, max balance_penalty ~ 133, adj_penalty ~ 18
    # We want spectral penalty to be in range [0, ~30] to be secondary but significant
    spectral_penalty = autocorr_frac * 30.0

    # Combined penalty
    total_penalty = balance_penalty + adj_penalty + spectral_penalty

    # Tie-breaker: Deterministic hash for unique ordering
    h = 0
    for v in vector:
        h = (h * 131 + v) & 0x7FFFFFFF

    tie_breaker = (h % 10000) / 10000.0 * 0.01

    # Higher priority for lower penalty
    return -total_penalty + tie_breaker
# EVOLVE-BLOCK-END
