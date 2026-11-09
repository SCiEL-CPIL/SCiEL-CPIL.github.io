"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    d = np.asarray(demands, dtype=float)
    n = len(d)

    # FFD assignment: sort by size descending, place each into first bin that fits.
    order = np.argsort(-d, kind="stable")
    bin_loads = []
    item_bin = np.full(n, -1, dtype=int)
    for idx in order:
        size = d[idx]
        placed = False
        for bi in range(len(bin_loads)):
            if bin_loads[bi] + size <= capacity:
                bin_loads[bi] += size
                item_bin[idx] = bi
                placed = True
                break
        if not placed:
            bin_loads.append(size)
            item_bin[idx] = len(bin_loads) - 1

    nbins = len(bin_loads)

    # Sparse compatibility: attractive only for pairs in the same FFD bin,
    # weighted by how tight (full) the bin is — tighter bins are "better" patterns.
    H = np.full((n, n), 1e-9, dtype=float)
    for bi in range(nbins):
        members = np.where(item_bin == bi)[0]
        if len(members) < 2:
            continue
        load = bin_loads[bi]
        # Tighter bin -> higher base attractiveness (smooth, in (0,1]).
        tight = load / capacity
        for i in members:
            for j in members:
                if i == j:
                    continue
                # Prefer placing a larger item next (continues a dense packing).
                size_j = d[j]
                # Reward transitions where j fits after i within this bin's budget.
                # Use remaining budget after i as a proxy for compatibility tightness.
                rem_after_i = capacity - (load - d[i])
                if size_j <= rem_after_i + 1e-12:
                    # Higher when bin is tight and j is large.
                    H[i, j] = tight * (0.5 + 0.5 * (size_j / capacity))
                else:
                    H[i, j] = 1e-9

    # Ensure finite and positive floor.
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H
# EVOLVE-BLOCK-END
