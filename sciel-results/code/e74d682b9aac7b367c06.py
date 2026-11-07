"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s kill

    n = len(demands)
    pair_sum = demands[:, None] + demands[None, :]
    gap = capacity - pair_sum

    # Initialize with very small value for invalid pairs
    H = np.full((n, n), 1e-9)

    # Valid pairs are those where both items fit together
    valid = gap >= 0

    # For each valid pair (i, j), compute how many other items could fit
    # into the remaining space gap[i,j] = capacity - d_i - d_j
    # This rewards pairs that leave "dead gaps" (too small for any remaining item)
    # or exact fits, rather than large unused voids.

    # Build a cumulative count and cumulative sum of demands by demand size
    demand_ints = np.round(demands).astype(int)
    max_demand = int(np.max(demand_ints))

    # Create a frequency array and cumulative sum of demands
    freq = np.zeros(max_demand + 1, dtype=np.int64)
    for d in demand_ints:
        freq[d] += 1

    # Cumulative sum: cumsum_count[k] = number of items with demand <= k
    cumsum_count = np.cumsum(freq)

    # Cumulative sum of demands: cumsum_demand[k] = sum of demands of items with demand <= k
    cumsum_demand = np.cumsum(freq * np.arange(max_demand + 1))

    # For each valid pair, compute metrics
    valid_indices = np.where(valid)
    i_idx = valid_indices[0]
    j_idx = valid_indices[1]
    gap_valid = gap[valid]

    # Clip gap to valid range for cumsum lookup
    gap_clipped = np.clip(gap_valid, 0, max_demand).astype(int)

    # Count of items with demand <= gap
    fit_counts = cumsum_count[gap_clipped]

    # Sum of demands of items with demand <= gap
    fit_demand_sums = cumsum_demand[gap_clipped]

    # Check if item i fits in gap[i,j]: d_i <= gap[i,j]
    i_fits = (2.0 * demands[i_idx] + demands[j_idx]) <= capacity
    # Check if item j fits in gap[i,j]: d_j <= gap[i,j]
    j_fits = (demands[i_idx] + 2.0 * demands[j_idx]) <= capacity

    # Adjust fit_counts: subtract 1 for each item that fits and is i or j
    adjusted_counts = fit_counts - i_fits.astype(np.int64) - j_fits.astype(np.int64)
    adjusted_counts = np.maximum(adjusted_counts, 0)

    # Adjust fit_demand_sums: subtract d_i if it fits, subtract d_j if it fits
    adjusted_demand_sums = fit_demand_sums - i_fits * demands[i_idx] - j_fits * demands[j_idx]
    adjusted_demand_sums = np.maximum(adjusted_demand_sums, 0.0)

    # The heuristic should reward pairs with FEWER fitting items and LESS total waste
    # in the residual. Use a combination:
    # 1. Inverse of (count + 1) for count-based penalty
    # 2. Inverse of (total_waste + 1) for waste-based penalty
    # 3. Bonus for near-exact fits (small gap)

    # Count-based component
    count_component = 1.0 / (adjusted_counts + 1.0)

    # Waste-based component: penalize large residual waste
    # Use inverse of (waste + 1) to reward small waste
    waste_component = 1.0 / (adjusted_demand_sums + 1.0)

    # Combine: weighted geometric mean to balance both signals
    # Geometric mean: (count_component * waste_component) ** 0.5
    combined = np.sqrt(count_component * waste_component)

    # Bonus for small gaps (near-exact fits)
    # Use a smooth penalty: exp(-gap / scale) where scale controls the width
    # gap=0 gets bonus 1.0, gap=10 gets ~0.37, gap=20 gets ~0.14
    gap_scale = 15.0
    gap_bonus = np.exp(-gap_valid / gap_scale)

    # Final heuristic: combined * (1 + gap_bonus) to boost near-exact fits
    H_valid = combined * (1.0 + gap_bonus)

    # Ensure all values are positive and finite
    H_valid = np.where(np.isfinite(H_valid) & (H_valid > 0), H_valid, 1e-9)

    H[valid] = H_valid

    # Check deadline
    if time.time() > deadline - 10:
        return H

    return H
# EVOLVE-BLOCK-END
