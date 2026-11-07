"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def _sigma_sieve(n):
    """Compute sigma(k) for k=1..n using a sieve."""
    sigma = [0] * (n + 1)
    for d in range(1, n + 1):
        for m in range(d, n + 1, d):
            sigma[m] += d
    return sigma

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    # Start with the known 3-way collision as fallback
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    # Search for collisions using sieve
    # k * sigma(k) <= 10^18 means k <= 10^18 / sigma(k)
    # sigma(k) >= k+1, so k <= 10^18 / (k+1), roughly k <= 10^9
    # But we also need k <= sqrt(M) = sqrt(k*sigma(k)) = sqrt(k)*sigma(k)^0.5
    # i.e., k^2 <= k * sigma(k) => k <= sigma(k), which is always true for k>=1
    # So the binding constraint is k*sigma(k) <= 10^18

    # For practical purposes, sigma(k) is usually around k * O(log log k)
    # so k can be up to roughly 10^18 / (10 * log log k) ~ 10^16 / log log k
    # But we need to actually compute sigma, so we're limited by memory/time
    # Let's try up to 2*10^6 first, then expand if time permits

    for limit in [2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000]:
        if time.time() > deadline - 20:
            break
        try:
            sigma = _sigma_sieve(limit)
            # Build map: v -> list of k
            groups = defaultdict(list)
            max_M = 10**18
            for k in range(1, limit + 1):
                v = k * sigma[k]
                if v > max_M:
                    continue
                # Check k <= sqrt(M): k^2 <= v = k*sigma(k) => k <= sigma(k)
                if k > sigma[k]:
                    continue
                groups[v].append(k)

            # Find best group
            for v, ks in groups.items():
                if len(ks) > best_count:
                    # Verify all k_i <= isqrt(v)
                    sq = math.isqrt(v)
                    valid_ks = [k for k in ks if k <= sq]
                    if len(valid_ks) > best_count:
                        best_count = len(valid_ks)
                        best_M = v
                        best_k = valid_ks[:1000]  # at most 1000
            del groups
            del sigma
        except (MemoryError, Exception):
            break

    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k[:1000]]

    return {'M': best_M, 'k': best_k}
# EVOLVE-BLOCK-END
