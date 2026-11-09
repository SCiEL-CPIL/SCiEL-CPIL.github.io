"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    d = np.asarray(demands, dtype=float)
    n = len(d)

    # Best-fit-decreasing style packing to identify good pairings.
    order = np.argsort(-d, kind="stable")
    bin_loads = []
    item_bin = np.full(n, -1, dtype=int)
    for idx in order:
        size = d[idx]
        # Best fit: choose bin with least remaining space that still fits.
        best_bi = -1
        best_rem = np.inf
        for bi in range(len(bin_loads)):
            rem = capacity - bin_loads[bi]
            if rem >= size and rem < best_rem:
                best_rem = rem
                best_bi = bi
        if best_bi >= 0:
            bin_loads[best_bi] += size
            item_bin[idx] = best_bi
        else:
            bin_loads.append(size)
            item_bin[idx] = len(bin_loads) - 1

    min_size = float(d.min())
    H = np.full((n, n), 1e-9, dtype=float)

    # Compute heuristic for all valid pairs (i, j) where d[i] + d[j] <= capacity
    # This creates a dense heuristic matrix, allowing the ACO to explore
    # combinations not explicitly found in the single BFD run.
    for i in range(n):
        si = d[i]
        for j in range(n):
            if i == j:
                continue
            sj = d[j]
            combined = si + sj
            if combined > capacity + 1e-12:
                continue
            
            residual = capacity - combined
            if residual < 0:
                continue

            # Structural fit: how close is residual to a multiple of min_size?
            if min_size > 0:
                k = int(round(residual / min_size))
                k = max(0, k)
                dist_to_k = abs(residual - k * min_size)
                dist_to_k1 = abs(residual - (k + 1) * min_size)
                min_dist = min(dist_to_k, dist_to_k1)
                struct_score = 1.0 / (1.0 + min_dist / max(min_size, 1e-9))
            else:
                struct_score = 1.0

            # Slack-based score: tighter is better (best-fit intuition)
            slack = residual
            slack_score = 1.0 / (1.0 + slack / 10.0)

            # Size bonus: prefer larger items to fill space quickly
            size_bonus = 0.5 + 0.5 * (sj / capacity)

            # Combine scores. We remove the 'tight' factor from BFD bins
            # because we are now evaluating all pairs globally.
            # The 'struct_score' and 'slack_score' provide the necessary discrimination.
            H[i, j] = struct_score * slack_score * size_bonus

    # Ensure finite and positive floor.
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H
# EVOLVE-BLOCK-END
