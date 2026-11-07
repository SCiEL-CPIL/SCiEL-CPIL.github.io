import numpy as np
import time

def _simulate_aco(demands, capacity, H, n_ants=20, n_iterations=30, seed=42):
    """
    Simulate the ACO process to evaluate the heuristic matrix H.
    Returns the mean number of bins used by the best ant in each iteration.
    """
    W = np.array(demands, dtype=float)
    n = len(W)
    
    # Precompute fit mask: can item j fit after item i?
    # Actually, the constraint is: current_bin_load + W[j] <= capacity
    # But the state is 'current item i' which implies the bin contains i and possibly others?
    # No, the task says: "placing item j right after item i in the same bin"
    # "An ant opens a bin, places a random fitting item, then repeatedly picks a fitting item..."
    # "probability proportional to pheromone[current, j] * heuristic[current, j]"
    # The state 'current' is the last item placed.
    # However, the constraint for fitting depends on the *total* load of the bin, not just the last item.
    # BUT, the heuristic matrix H is fixed. The pheromone is updated based on bins.
    # The probability calculation uses H[current, j].
    # If H[current, j] is 1e-9 (or very small) when j doesn't fit, the ant won't pick it.
    # So the heuristic MUST encode the fitting constraint relative to the *current bin state*.
    # But the heuristic is computed ONCE before the colony runs. It cannot depend on the dynamic bin load.
    # This implies the heuristic H[i,j] is a static property of the pair (i,j).
    # How can a static heuristic enforce the capacity constraint?
    # It can't directly. The ant must check if the item fits in the current bin.
    # The prompt says: "repeatedly picks a fitting item with probability proportional to..."
    # This implies that if an item does NOT fit, it is not a "fitting item" and thus not considered.
    # So the probability is only over items that fit.
    # Therefore, the heuristic H[i,j] only needs to be attractive for valid pairs.
    # If H[i,j] is large for pairs that fit, and small for pairs that don't, it helps.
    # But since the ant filters for "fitting items" first, H[i,j] for non-fitting items is irrelevant to the choice,
    # EXCEPT that the code might not filter correctly if H is not zeroed out?
    # "Heuristic entries that are not positive become 1e-9"
    # If the ant logic is:
    #   candidates = [j for j in items if current_load + W[j] <= capacity]
    #   probs = [P[current, j] * H[current, j] for j in candidates]
    # Then H[current, j] for j not in candidates is ignored.
    # So H[i,j] should be high for good pairs (i,j) regardless of whether they fit in a specific bin state,
    # because the "fitting" check is done separately.
    # However, if H[i,j] is high for a pair (i,j) that *never* fits (W[i]+W[j] > capacity), it doesn't matter.
    # But if W[i]+W[j] <= capacity, it might fit in a bin containing just i.
    # So a good heuristic should prefer pairs (i,j) that are "compatible" in a way that leads to good packing.
    
    # Let's re-read carefully: "returns an item-by-item matrix of attractiveness for placing item j right after item i in the same bin"
    # "Heuristic entries that are not positive become 1e-9"
    # "Ant... picks a fitting item with probability proportional to pheromone[current, j] * heuristic[current, j]"
    
    # Key insight: The heuristic is static. The "fitting" constraint is dynamic.
    # The heuristic should guide the ant towards items that complete bins efficiently.
    # A common heuristic is "Best Fit" or "Worst Fit" logic encoded statically.
    # Since we don't know the bin load, we can only use item sizes.
    # Preference for larger items (Largest Fit) is a good baseline.
    
    # Let's simulate to see what works.
    
    rng = np.random.default_rng(seed)
    
    # Pheromone initialization
    P = np.ones((n, n))
    
    best_bins_per_iter = []
    
    for it in range(n_iterations):
        min_bins_this_iter = float('inf')
        
        for ant in range(n_ants):
            bins = []
            current_bin_load = 0
            current_item = -1  # No item placed yet
            
            # Place first item randomly
            # "places a random fitting item"
            # Any item fits in an empty bin.
            # Random choice.
            if n > 0:
                first_idx = rng.integers(n)
                current_bin_load = W[first_idx]
                current_item = first_idx
                used = np.zeros(n, dtype=bool)
                used[first_idx] = True
            else:
                continue
            
            # Repeatedly pick fitting item
            while True:
                # Find fitting items
                # An item j fits if current_bin_load + W[j] <= capacity
                # And j is not already used.
                fits_mask = (current_bin_load + W <= capacity) & (~used)
                
                if not np.any(fits_mask):
                    # Open new bin
                    bins.append(current_bin_load)
                    # Reset
                    current_bin_load = 0
                    current_item = -1
                    # Pick a random fitting item for the new bin
                    # All items are fitting in an empty bin.
                    # But we must pick an unused one.
                    remaining = np.where(~used)[0]
                    if len(remaining) == 0:
                        break
                    first_idx = rng.choice(remaining)
                    current_bin_load = W[first_idx]
                    current_item = first_idx
                    used[first_idx] = True
                    continue
                
                # Calculate probabilities
                # P[current_item, j] * H[current_item, j]
                # Note: current_item is the last item placed.
                # If current_item is -1 (shouldn't happen in loop after first placement), handle carefully.
                
                # Get pheromone and heuristic for candidate j
                # We only consider j where fits_mask is True.
                
                # Optimization: Vectorized probability calculation
                # P_row = P[current_item, :]
                # H_row = H[current_item, :]
                # scores = P_row * H_row
                # scores[~fits_mask] = 0
                # probs = scores / scores.sum()
                
                # But we need to handle the case where current_item is -1?
                # In the loop, current_item is always set.
                
                if current_item == -1:
                    # This case is handled in the "new bin" branch above.
                    # So here current_item >= 0.
                    pass
                
                P_row = P[current_item, :]
                H_row = H[current_item, :]
                scores = P_row * H_row
                
                # Mask out non-fitting items
                scores = np.where(fits_mask, scores, 0.0)
                
                total_score = scores.sum()
                if total_score == 0:
                    # Fallback: pick random fitting item
                    candidates = np.where(fits_mask)[0]
                    next_idx = rng.choice(candidates)
                else:
                    probs = scores / total_score
                    # Sample
                    next_idx = rng.choice(n, p=probs)
                
                used[next_idx] = True
                current_bin_load += W[next_idx]
                current_item = next_idx
            
            # Finalize last bin
            if current_bin_load > 0:
                bins.append(current_bin_load)
            
            if len(bins) < min_bins_this_iter:
                min_bins_this_iter = len(bins)
        
        best_bins_per_iter.append(min_bins_this_iter)
        
        # Pheromone update
        # "pheromone is multiplied by 0.9 and every entry gains 1/(fewest bins of that iteration)"
        P = P * 0.9 + 1.0 / min_bins_this_iter
    
    return np.mean(best_bins_per_iter)


def _tune_heuristic(demands, capacity):
    """
    Tune the heuristic parameters using a simple search.
    We will try a few different heuristic forms.
    """
    W = np.array(demands, dtype=float)
    n = len(W)
    
    # Candidate heuristic generators
    def gen_h1(W, cap):
        # Largest Fit: H[i,j] = W[j]^2 if W[i]+W[j] <= cap else 1e-9
        pair_sums = W[:, None] + W[None, :]
        fit_mask = pair_sums <= cap
        H = np.where(fit_mask, W[None, :] ** 2, 1e-9)
        return H
    
    def gen_h2(W, cap):
        # Inverse Gap: Prefer items that fill the bin more completely?
        # H[i,j] = 1 / (cap - (W[i]+W[j])) if W[i]+W[j] <= cap else 1e-9
        # This favors tight fits.
        pair_sums = W[:, None] + W[None, :]
        fit_mask = pair_sums <= cap
        gap = cap - pair_sums
        # Avoid division by zero if gap is 0? 1/0 is inf.
        # If gap is 0, it's a perfect fit, should be very attractive.
        # Let's use 1 / (gap + 1) to be safe, or handle 0 separately.
        H = np.where(fit_mask, 1.0 / (gap + 1.0), 1e-9)
        # Scale to be comparable to other heuristics?
        # The magnitude doesn't matter much as long as relative order is preserved,
        # but it interacts with pheromone.
        # Let's normalize H to have mean 1 for fitting entries.
        fitting_vals = H[fit_mask]
        if len(fitting_vals) > 0:
            mean_val = np.mean(fitting_vals)
            if mean_val > 0:
                H[fit_mask] = H[fit_mask] / mean_val
        return H
    
    def gen_h3(W, cap):
        # Hybrid: W[j] / gap
        pair_sums = W[:, None] + W[None, :]
        fit_mask = pair_sums <= cap
        gap = cap - pair_sums
        H = np.where(fit_mask, W[None, :] / (gap + 1.0), 1e-9)
        fitting_vals = H[fit_mask]
        if len(fitting_vals) > 0:
            mean_val = np.mean(fitting_vals)
            if mean_val > 0:
                H[fit_mask] = H[fit_mask] / mean_val
        return H

    candidates = [gen_h1, gen_h2, gen_h3]
    
    best_score = float('inf')
    best_H = None
    
    for gen in candidates:
        H = gen(W, capacity)
        # Simulate
        score = _simulate_aco(demands, capacity, H, n_ants=20, n_iterations=30, seed=42)
        if score < best_score:
            best_score = score
            best_H = H
    
    return best_H


def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    """
    start_time = time.time()
    deadline = start_time + 300  # 300s limit, keep margin
    
    W = np.array(demands, dtype=float)
    n = len(W)
    
    # Fallback heuristic: Largest Fit
    def default_heuristic(W, cap):
        pair_sums = W[:, None] + W[None, :]
        fit_mask = pair_sums <= cap
        H = np.where(fit_mask, W[None, :] ** 2, 1e-9)
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        return H
    
    H = default_heuristic(W, capacity)
    
    # Try to tune if time permits
    # Tuning involves simulation which takes time.
    # Let's check how long simulation takes.
    # If it's fast, we can do it.
    
    if time.time() < deadline - 60:  # 60s margin
        try:
            H_tuned = _tune_heuristic(demands, capacity)
            if H_tuned is not None:
                H = H_tuned
        except Exception:
            pass
    
    # Final safety check
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
