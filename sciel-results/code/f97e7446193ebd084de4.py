# EVOLVE-BLOCK-START
import numpy as np
import time


def _run_aco_single(demands, capacity, heuristic, n_ants, n_iters, seed, time_limit=280.0):
    """
    Run one ACO simulation and return the fewest bins used by any ant across all iterations.
    """
    n = len(demands)
    tau = np.ones((n, n))
    best_overall = n  # worst case: each item in its own bin

    rng = np.random.default_rng(seed)
    start_time = time.time()
    # Precompute sorted demands for faster fitting checks
    d_float = demands.astype(np.float64)

    for it in range(n_iters):
        # Time guard: stop if we're running out of time
        if time.time() - start_time > time_limit:
            break

        best_ant_bins = n
        best_ant_path = None  # list of (i, j) edges for the best ant this iteration

        for ant in range(n_ants):
            if time.time() - start_time > time_limit:
                break
            used = np.zeros(n, dtype=bool)
            bins = 0
            current_sum = 0.0
            current_item = -1  # no item placed yet in current bin
            path = []

            # Place first item randomly
            available = np.where(~used)[0]
            first = available[rng.integers(len(available))]
            used[first] = True
            current_sum = d_float[first]
            current_item = first
            bins = 1

            # Now place remaining items
            while True:
                if time.time() - start_time > time_limit:
                    break
                # Find all unused items that fit
                remaining = np.where(~used)[0]
                if len(remaining) == 0:
                    break

                # Filter to those that fit
                slack = capacity - current_sum
                fitting = remaining[d_float[remaining] <= slack]

                if len(fitting) == 0:
                    # Open new bin
                    bins += 1
                    current_sum = 0.0
                    current_item = -1
                    available = np.where(~used)[0]
                    first = available[rng.integers(len(available))]
                    used[first] = True
                    current_sum = d_float[first]
                    current_item = first
                    continue

                # Compute probabilities: tau[current_item, j] * heuristic[current_item, j]
                probs = tau[current_item, fitting] * heuristic[current_item, fitting]
                # Ensure no zeros cause issues; if all are zero, pick uniformly
                total = probs.sum()
                if total <= 0 or not np.isfinite(total):
                    idx = rng.integers(len(fitting))
                else:
                    # Normalize for stability
                    max_p = np.max(probs)
                    if max_p > 0:
                        probs = probs / max_p
                    probs = probs / probs.sum()
                    # Use cumulative for sampling
                    cum = np.cumsum(probs)
                    cum[-1] = 1.0  # avoid floating point issues
                    r = rng.random()
                    idx = np.searchsorted(cum, r)
                    idx = min(idx, len(fitting) - 1)

                chosen = fitting[idx]
                used[chosen] = True
                path.append((current_item, chosen))
                current_sum += d_float[chosen]
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
            # Also reinforce diagonal slightly
            tau[np.diag_indices(n)] += gain * 0.1

    return best_overall


def heuristics(demands, capacity):
    """
    Compute the heuristic matrix.
    
    FFD-adjacency heuristic with size bias:
    - Pre-compute a First-Fit-Decreasing bin assignment
    - Assign high attractiveness to pairs of items that end up in the same FFD bin
    - Within same bin, modulate by tightness AND item size (larger items preferred)
    - Items in different FFD bins get low (but positive) attractiveness
    """
    n = len(demands)
    d = demands.astype(np.float64)
    
    # Sort items by size descending for FFD
    sorted_indices = np.argsort(-d)
    
    # Perform FFD assignment
    bins = []  # list of (bin_sum, [item_indices_in_original_order])
    item_to_bin = np.zeros(n, dtype=int)
    
    for idx in sorted_indices:
        placed = False
        for b_idx in range(len(bins)):
            if bins[b_idx][0] + d[idx] <= capacity:
                bins[b_idx] = (bins[b_idx][0] + d[idx], bins[b_idx][1] + [idx])
                item_to_bin[idx] = b_idx
                placed = True
                break
        if not placed:
            bins.append((d[idx], [idx]))
            item_to_bin[idx] = len(bins) - 1
    
    # Build heuristic matrix
    same_bin = (item_to_bin[:, None] == item_to_bin[None, :])
    
    # Pair sums and fit mask
    pair = d[:, None] + d[None, :]
    mask = pair <= capacity
    
    # Tight packing score: prefer pairs that fill the bin well
    # Use a smoother formulation: (pair/capacity)^alpha * exp(-beta * slack)
    ratio = np.where(mask, pair / capacity, 0.0)
    slack = np.where(mask, capacity - pair, capacity)
    
    # Smaller alpha for smoother landscape, moderate beta
    alpha = 4.0
    beta = 0.05
    base = ratio ** alpha
    gap_penalty = np.exp(-beta * slack)
    tight_score = base * gap_penalty
    
    # Size bias: larger items get higher attractiveness (mimics BFD)
    # Normalize sizes to [0, 1] range for the bias
    min_d = d.min()
    max_d = d.max()
    if max_d > min_d:
        size_norm = (d - min_d) / (max_d - min_d)
    else:
        size_norm = np.ones(n) * 0.5
    
    # Size bias for the destination item j (prefer placing larger items)
    size_bias_j = 1.0 + 0.5 * size_norm[None, :]  # shape (1, n), broadcast over i
    
    # Apply size bias to tight_score for fitting pairs
    tight_score_biased = tight_score * size_bias_j
    
    # Heuristic: 
    # - Same bin AND fits: tight_score_biased (high, favors tight packing + larger items)
    # - Same bin AND doesn't fit: moderate value
    # - Different bin: very low value
    same_bin_fits = same_bin & mask
    same_bin_no_fit = same_bin & ~mask
    diff_bin = ~same_bin
    
    heuristic = np.zeros((n, n))
    heuristic[same_bin_fits] = tight_score_biased[same_bin_fits]
    heuristic[same_bin_no_fit] = 0.1
    heuristic[diff_bin] = 1e-9
    
    # Ensure diagonal is positive
    heuristic[np.diag_indices(n)] = 1e-9
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic


# EVOLVE-BLOCK-END
