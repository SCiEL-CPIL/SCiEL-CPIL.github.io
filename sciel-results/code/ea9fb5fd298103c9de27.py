"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes a heuristic matrix using diverse FFD/BFD/WFD simulations
    to learn co-occurrence patterns, combined with smooth gap utility.
    """
    start_time = time.time()
    time_limit = 280.0  # Safety margin before 300s kill

    n = len(demands)
    demands = np.asarray(demands, dtype=float)

    # Pair sums and validity
    pair_sums = demands[:, None] + demands[None, :]
    valid = pair_sums <= capacity
    remaining = capacity - pair_sums
    min_item = np.min(demands)

    # Co-occurrence count matrix
    cooccur = np.zeros((n, n), dtype=np.float64)

    # Sorted indices by demand descending
    sorted_idx = np.argsort(-demands)
    sorted_demands = demands[sorted_idx]

    def run_ffd(order_idx, bins_list=None):
        """Run FFD with given item order, return bins and update cooccur."""
        bins = []  # List of [current_sum, [item_indices]]
        for idx in order_idx:
            item_size = demands[idx]
            placed = False
            for b in bins:
                if b[0] + item_size <= capacity:
                    b[0] += item_size
                    for other in b[1]:
                        cooccur[idx, other] += 1
                        cooccur[other, idx] += 1
                    b[1].append(idx)
                    placed = True
                    break
            if not placed:
                bins.append([item_size, [idx]])
        return bins

    def run_bfd(order_idx):
        """Run Best-Fit-Decreasing: place in bin with least remaining space that fits."""
        bins = []  # List of [current_sum, [item_indices]]
        for idx in order_idx:
            item_size = demands[idx]
            best_b = -1
            best_remaining = capacity + 1
            for b_idx, b in enumerate(bins):
                rem = capacity - b[0]
                if rem >= item_size and rem < best_remaining:
                    best_remaining = rem
                    best_b = b_idx
            if best_b >= 0:
                b = bins[best_b]
                b[0] += item_size
                for other in b[1]:
                    cooccur[idx, other] += 1
                    cooccur[other, idx] += 1
                b[1].append(idx)
            else:
                bins.append([item_size, [idx]])
        return bins

    def run_wfd(order_idx):
        """Run Worst-Fit-Decreasing: place in bin with most remaining space that fits."""
        bins = []
        for idx in order_idx:
            item_size = demands[idx]
            best_b = -1
            best_remaining = -1
            for b_idx, b in enumerate(bins):
                rem = capacity - b[0]
                if rem >= item_size and rem > best_remaining:
                    best_remaining = rem
                    best_b = b_idx
            if best_b >= 0:
                b = bins[best_b]
                b[0] += item_size
                for other in b[1]:
                    cooccur[idx, other] += 1
                    cooccur[other, idx] += 1
                b[1].append(idx)
            else:
                bins.append([item_size, [idx]])
        return bins

    # Run diverse packing algorithms
    # Phase 1: Standard FFD/BFD/WFD
    run_ffd(sorted_idx)
    run_bfd(sorted_idx)
    run_wfd(sorted_idx)

    # Phase 2: Randomized perturbations for diversity
    rng = np.random.RandomState(42)
    num_random_runs = 0
    max_random_runs = 3000

    for run in range(max_random_runs):
        if time.time() - start_time > time_limit * 0.6:
            break

        # Create a perturbed order: mostly descending, but swap some small items
        order = sorted_idx.copy()
        # Swap random pairs of items in the lower half (smaller items)
        half = n // 2
        for _ in range(rng.randint(1, 6)):
            i = rng.randint(half, n)
            j = rng.randint(half, n)
            if i != j:
                order[i], order[j] = order[j], order[i]

        # Randomly choose FFD, BFD, or WFD
        choice = rng.randint(3)
        if choice == 0:
            run_ffd(order)
        elif choice == 1:
            run_bfd(order)
        else:
            run_wfd(order)

        num_random_runs += 1

    # Phase 3: If time remains, run more with different perturbation strategies
    if time.time() - start_time < time_limit * 0.75:
        for run in range(2000):
            if time.time() - start_time > time_limit * 0.85:
                break
            # Reverse some segments
            order = sorted_idx.copy()
            seg_start = rng.randint(0, n - 10)
            seg_end = min(seg_start + rng.randint(5, 20), n)
            order[seg_start:seg_end] = order[seg_start:seg_end][::-1]

            choice = rng.randint(3)
            if choice == 0:
                run_ffd(order)
            elif choice == 1:
                run_bfd(order)
            else:
                run_wfd(order)

    # Normalize co-occurrence: use log scaling to emphasize rare pairs
    # Add 1 to avoid log(0), then log-scale
    cooccur_smooth = np.log1p(cooccur)
    max_cooccur = np.max(cooccur_smooth)
    if max_cooccur > 0:
        cooccur_norm = cooccur_smooth / max_cooccur
    else:
        cooccur_norm = np.zeros_like(cooccur_smooth)

    # --- Gap-based scoring with smooth functions ---
    # For valid pairs, score based on how tightly they pack
    # Use a smooth function: higher score for smaller remaining space
    # remaining is in [0, capacity]
    # Score: 1 / (1 + remaining * scale) where scale adjusts sensitivity
    scale = 0.05  # Tuning parameter
    gap_score = np.zeros((n, n), dtype=float)
    gap_score[valid] = 1.0 / (1.0 + remaining[valid] * scale)

    # Boost perfect fits
    perfect_fit = valid & (remaining < 1e-9)
    gap_score[perfect_fit] = 5.0

    # Penalize dead gaps (remaining > 0 but < min_item)
    dead_gap = valid & (remaining > 1e-9) & (remaining < min_item)
    gap_score[dead_gap] = 0.1

    # --- Combine co-occurrence and gap scores ---
    # Weighted combination
    w_cooccur = 0.6
    w_gap = 0.4

    eta = np.full((n, n), 1e-9, dtype=float)
    eta[valid] = w_cooccur * cooccur_norm[valid] + w_gap * gap_score[valid]

    # Ensure invalid pairs stay at minimum
    eta[~valid] = 1e-9

    # Diagonal: self-pairing is not meaningful
    np.fill_diagonal(eta, 1e-9)

    # Ensure all values are positive and finite
    eta[eta <= 0] = 1e-9
    eta[~np.isfinite(eta)] = 1e-9

    # Final safety: ensure no NaN or Inf
    if not np.all(np.isfinite(eta)):
        eta = np.where(np.isfinite(eta), eta, 1e-9)
        eta[eta <= 0] = 1e-9

    return eta
# EVOLVE-BLOCK-END
