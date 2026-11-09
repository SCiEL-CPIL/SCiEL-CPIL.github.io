# EVOLVE-BLOCK-START
import numpy as np
import time


def _run_aco_single(demands, capacity, heuristic, n_ants, n_iters, seed):
    """
    Run one ACO simulation and return the fewest bins used by any ant across all iterations.
    """
    n = len(demands)
    tau = np.ones((n, n))
    best_overall = n  # worst case: each item in its own bin

    rng = np.random.default_rng(seed)

    # Precompute which items fit with each other (for quick reference)
    # Actually, we check capacity - current_sum >= demand[j] at runtime.

    for it in range(n_iters):
        best_ant_bins = n
        best_ant_path = None  # list of (i, j) edges for the best ant this iteration

        for ant in range(n_ants):
            used = np.zeros(n, dtype=bool)
            bins = 0
            current_sum = 0.0
            current_item = -1  # no item placed yet in current bin
            path = []

            # Place first item randomly
            # Pick a random unused item
            available = np.where(~used)[0]
            first = available[rng.integers(len(available))]
            used[first] = True
            current_sum = demands[first]
            current_item = first
            bins = 1

            # Now place remaining items
            while True:
                # Find all unused items that fit
                remaining = np.where(~used)[0]
                if len(remaining) == 0:
                    break

                # Filter to those that fit
                slack = capacity - current_sum
                fitting = remaining[demands[remaining] <= slack]

                if len(fitting) == 0:
                    # Open new bin
                    bins += 1
                    current_sum = 0.0
                    current_item = -1
                    # Pick a random fitting item for the new bin (all fit since bin is empty)
                    # Actually, any unused item fits in an empty bin
                    available = np.where(~used)[0]
                    first = available[rng.integers(len(available))]
                    used[first] = True
                    current_sum = demands[first]
                    current_item = first
                    continue

                # Compute probabilities: tau[current_item, j] * heuristic[current_item, j]
                probs = tau[current_item, fitting] * heuristic[current_item, fitting]
                # Ensure no zeros cause issues; if all are zero, pick uniformly
                total = probs.sum()
                if total <= 0 or not np.isfinite(total):
                    idx = rng.integers(len(fitting))
                else:
                    probs = probs / total
                    # Use cumulative for sampling
                    cum = np.cumsum(probs)
                    cum[-1] = 1.0  # avoid floating point issues
                    r = rng.random()
                    idx = np.searchsorted(cum, r)
                    idx = min(idx, len(fitting) - 1)

                chosen = fitting[idx]
                used[chosen] = True
                path.append((current_item, chosen))
                current_sum += demands[chosen]
                current_item = chosen

            if bins < best_ant_bins:
                best_ant_bins = bins
                best_ant_path = path

        if best_ant_bins < best_overall:
            best_overall = best_ant_bins

        # Pheromone update
        tau *= 0.9
        if best_ant_bins > 0:
            gain = 1.0 / best_ant_bins
            if best_ant_path is not None:
                for i, j in best_ant_path:
                    tau[i, j] += gain
            # Also, should we add to the initial placement? The path doesn't include the first item of each bin.
            # The problem says "placing item j right after item i". The first item in a bin has no predecessor.
            # So we only update edges that were explicitly traversed.

    return best_overall


def heuristics(demands, capacity):
    """
    Compute the heuristic matrix.
    We use a 'Best Fit Decreasing' inspired heuristic: prefer pairs that leave less slack,
    but also consider the absolute size to avoid placing two very small items together if a
    larger item could fit better.
    
    Actually, the standard ACO for bin packing often uses a heuristic that favors tight packing.
    Let's try: h[i,j] = 1 / (slack + epsilon) if it fits, where slack = capacity - (d[i] + d[j]).
    Or h[i,j] = (d[i] + d[j]) / capacity if it fits.
    
    Based on the observed improvement with ratio**4, let's stick to a power of the ratio.
    ratio = (d[i] + d[j]) / capacity.
    Higher ratio means tighter fit.
    """
    n = len(demands)
    pair = demands[:, None] + demands[None, :]
    mask = pair <= capacity
    
    # Use a high power to strongly favor tight fits
    ratio = np.where(mask, pair / capacity, 0)
    heuristic = np.where(mask, ratio ** 6, 1e-9)
    
    return heuristic


# EVOLVE-BLOCK-END
