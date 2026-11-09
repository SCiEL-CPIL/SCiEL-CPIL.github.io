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
            # Pick a random unused item
            available = np.where(~used)[0]
            first = available[rng.integers(len(available))]
            used[first] = True
            current_sum = demands[first]
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

    return best_overall


def heuristics(demands, capacity):
    """
    Compute the heuristic matrix.
    
    Complementarity-based heuristic:
    - Rewards pairs that leave residual capacity as a near-integer multiple of the
      smallest item size, promoting structural fit over simple best-fit.
    - Combines with a smooth ratio-based attraction for overall bin utilization.
    
    The heuristic is: h[i,j] = ratio^alpha * exp(-beta * slack) * complementarity_score
    where complementarity_score measures how well the slack aligns with item sizes.
    """
    n = len(demands)
    pair = demands[:, None] + demands[None, :]
    mask = pair <= capacity
    
    # Base ratio: how full the bin would be
    ratio = np.where(mask, pair / capacity, 0.0)
    
    # Slack: remaining space (lower is better)
    slack = np.where(mask, capacity - pair, capacity)
    
    # Complementarity score: reward slack that is a near-integer multiple of smallest item
    min_item = np.min(demands)
    if min_item > 0:
        # For each pair (i,j), compute how well slack aligns with multiples of min_item
        # We want slack to be close to k * min_item for some integer k >= 0
        # This means items can be added later to fill the bin nicely
        # Compute distance to nearest multiple of min_item
        k_float = slack / min_item
        k_round = np.round(k_float)
        dist_to_multiple = np.abs(slack - k_round * min_item)
        
        # Normalize: smaller distance is better
        # Use exponential decay based on normalized distance
        max_dist = min_item / 2.0  # half the smallest item is the max meaningful distance
        complementarity = np.exp(-2.0 * dist_to_multiple / max_dist)
        
        # Also consider: if slack is very small, that's great (tight fit)
        # If slack is large but aligned, that's also good (room for more items)
        # Blend: prefer small slack, but reward alignment
        slack_weight = np.exp(-0.01 * slack)
        complementarity_score = 0.5 * slack_weight + 0.5 * complementarity
    else:
        complementarity_score = np.ones_like(slack)
    
    # Combine all components
    alpha = 3.0
    beta = 0.015
    
    base = ratio ** alpha
    gap_penalty = np.exp(-beta * slack)
    
    heuristic = np.where(mask, base * gap_penalty * complementarity_score, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic


# EVOLVE-BLOCK-END
