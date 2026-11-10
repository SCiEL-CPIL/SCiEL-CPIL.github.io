import numpy as np
import time

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i
    in the same bin. Larger is more attractive.
    
    Strategy:
    1. Run a fast greedy First-Fit-Decreasing (FFD) algorithm to get a good packing.
    2. Run a local search (swap-based) to further improve the packing (reduce bins or waste).
    3. Construct the heuristic matrix based on which items were packed together in the final solution.
       - Pairs in the same bin get high attractiveness.
       - Pairs in different bins get low attractiveness (but still valid if they fit).
       - Invalid pairs (sum > capacity) get 1e-9.
    4. This "learned" heuristic guides the ACO to reproduce the structure of the good solution found by FFD+LS.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 300s limit
    
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    # 1. Greedy First-Fit-Decreasing (FFD)
    indices = np.argsort(d)[::-1]  # Sort indices by demand descending
    sorted_d = d[indices]
    
    bins = []  # List of lists of original indices
    bin_sums = []
    
    for idx in indices:
        placed = False
        for b in range(len(bins)):
            if bin_sums[b] + d[idx] <= capacity:
                bins[b].append(idx)
                bin_sums[b] += d[idx]
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_sums.append(d[idx])
    
    # 2. Local Search: Try swapping items between bins to reduce total waste or enable bin removal
    # We will iterate a few times, checking if a swap improves the objective (fewer bins, or less total waste)
    # Objective: minimize number of bins, then minimize total waste (sum of (capacity - bin_sum) for non-empty bins)
    
    def get_objective(bins, bin_sums):
        num_bins = len(bins)
        total_waste = sum(capacity - s for s in bin_sums)
        return (num_bins, total_waste)
    
    current_obj = get_objective(bins, bin_sums)
    
    # Simple local search: try swapping any two items from different bins
    # To keep it fast, we limit the number of iterations and check time
    max_ls_iterations = 5000
    ls_iter = 0
    improved = True
    while improved and ls_iter < max_ls_iterations:
        if time.time() > deadline - 10:
            break
        improved = False
        ls_iter += 1
        
        # Iterate over all pairs of bins
        for b1 in range(len(bins)):
            for b2 in range(b1 + 1, len(bins)):
                # Try swapping item i from bin1 with item j from bin2
                for i_idx in bins[b1]:
                    for j_idx in bins[b2]:
                        # Check if swap is valid
                        # New sum1 = sum1 - d[i] + d[j]
                        # New sum2 = sum2 - d[j] + d[i]
                        new_sum1 = bin_sums[b1] - d[i_idx] + d[j_idx]
                        new_sum2 = bin_sums[b2] - d[j_idx] + d[i_idx]
                        
                        if new_sum1 <= capacity and new_sum2 <= capacity:
                            # Perform swap
                            bins[b1].remove(i_idx)
                            bins[b2].remove(j_idx)
                            bins[b1].append(j_idx)
                            bins[b2].append(i_idx)
                            old_sum1 = bin_sums[b1]
                            old_sum2 = bin_sums[b2]
                            bin_sums[b1] = new_sum1
                            bin_sums[b2] = new_sum2
                            
                            new_obj = get_objective(bins, bin_sums)
                            if new_obj < current_obj:
                                current_obj = new_obj
                                improved = True
                            else:
                                # Revert
                                bins[b1].remove(j_idx)
                                bins[b2].remove(i_idx)
                                bins[b1].append(i_idx)
                                bins[b2].append(j_idx)
                                bin_sums[b1] = old_sum1
                                bin_sums[b2] = old_sum2
                                # If no improvement in this pair, break early to save time?
                                # No, because another swap in same bins might work. But for speed, we can break inner loops if we found an improvement.
                                if improved:
                                    break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    # 3. Construct Heuristic Matrix
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # Compute pair sums for validity
    pair_sums = d[:, None] + d[None, :]
    valid = pair_sums <= capacity
    
    # Base value for valid pairs that are NOT in the same bin in the solution
    # We want to encourage the ACO to find the solution we found, so same-bin pairs should be high.
    # However, if the ACO is stuck, it might need to explore. 
    # Let's set same-bin pairs to a high value, and other valid pairs to a lower value.
    
    # Create a matrix indicating if two items are in the same bin
    same_bin = np.zeros((n, n), dtype=bool)
    for b in bins:
        for i in b:
            for j in b:
                same_bin[i, j] = True
    
    # Initialize eta with low value for all valid pairs
    eta[valid] = 1.0
    
    # Boost pairs that are in the same bin in the FFD+LS solution
    # The boost should be significant.
    # Also, consider the "tightness" of the bin they are in?
    # Let's just use a strong boost for co-occurrence.
    eta[same_bin & valid] = 100.0
    
    # Further refinement: If a bin is "full" (high utilization), boost those pairs even more?
    # Or simply, ensure that pairs that allow a bin to be closed are boosted.
    # The current approach is a structural hint.
    
    # Ensure no non-finite or non-positive values
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
