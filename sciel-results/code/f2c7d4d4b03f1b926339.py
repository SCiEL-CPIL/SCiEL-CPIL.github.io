import numpy as np
import time

def _ffd_assignment(demands, capacity):
    """
    Perform First-Fit Decreasing bin packing.
    Returns an array of bin indices for each item.
    """
    n = len(demands)
    # Sort indices by demand descending
    sorted_indices = np.argsort(-demands)
    
    bins = []  # List of lists of items in each bin
    bin_loads = [] # Current load of each bin
    item_bin = np.full(n, -1, dtype=int)
    
    for idx in sorted_indices:
        size = demands[idx]
        placed = False
        for b_idx in range(len(bins)):
            if bin_loads[b_idx] + size <= capacity:
                bins[b_idx].append(idx)
                bin_loads[b_idx] += size
                item_bin[idx] = b_idx
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_loads.append(size)
            item_bin[idx] = len(bins) - 1
            
    return item_bin

def _build_ffd_prior(demands, capacity):
    """
    Build a heuristic matrix based on FFD bin assignments.
    High values for pairs in the same bin.
    """
    n = len(demands)
    item_bin = _ffd_assignment(demands, capacity)
    
    # Matrix where entry (i,j) is 1 if i and j are in the same bin
    same_bin = (item_bin[:, None] == item_bin[None, :]).astype(float)
    
    # Exclude self-pairs? The ACO logic picks j != current.
    # The heuristic matrix is indexed by [current, j].
    # If i == j, it doesn't matter much, but let's zero it out to be clean.
    np.fill_diagonal(same_bin, 0)
    
    return same_bin

def _fast_aco_eval(demands, capacity, heuristic, n_ants=5, n_iters=5, seed=42):
    """
    Fast evaluation of a heuristic matrix using a simplified ACO.
    Returns the best number of bins found.
    """
    n = len(demands)
    rng = np.random.default_rng(seed)
    pheromone = np.ones((n, n))
    best_bins = n
    
    # Precompute fit mask for speed
    pair_sum = demands[:, None] + demands[None, :]
    fits_mask = (pair_sum <= capacity)
    
    for _ in range(n_iters):
        iteration_best = n
        for _ in range(n_ants):
            placed = np.zeros(n, dtype=bool)
            bins = 0
            current = -1
            
            while not placed.all():
                if current == -1:
                    # Open new bin: pick random fitting item
                    available = np.where(~placed)[0]
                    if len(available) == 0: break
                    current = available[rng.integers(len(available))]
                    placed[current] = True
                    bins += 1
                else:
                    # Pick next item
                    available = np.where(~placed)[0]
                    if len(available) == 0: break
                    
                    # Filter by fit
                    fit_indices = available[fits_mask[current, available]]
                    if len(fit_indices) == 0:
                        current = -1
                        continue
                    
                    # Calculate probabilities
                    # pheromone[current, j] * heuristic[current, j]
                    probs = pheromone[current, fit_indices] * heuristic[current, fit_indices]
                    probs = np.where(probs > 0, probs, 1e-9)
                    
                    total = probs.sum()
                    if total <= 0:
                        current = fit_indices[rng.integers(len(fit_indices))]
                    else:
                        r = rng.random()
                        cumsum = np.cumsum(probs) / total
                        idx = np.searchsorted(cumsum, r)
                        if idx >= len(fit_indices):
                            idx = len(fit_indices) - 1
                        current = fit_indices[idx]
                    placed[current] = True
                    
            iteration_best = min(iteration_best, bins)
            
        # Update pheromone
        pheromone *= 0.9
        if iteration_best > 0:
            pheromone += 1.0 / iteration_best
            
        best_bins = min(best_bins, iteration_best)
        
    return best_bins

def heuristics(demands, capacity):
    start = time.time()
    deadline = 300.0
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # 1. FFD Prior
    ffd_prior = _build_ffd_prior(demands, capacity)
    
    # 2. Tightness Prior
    pair_sum = demands[:, None] + demands[None, :]
    fits = (pair_sum <= capacity)
    remaining = capacity - pair_sum
    # For fitting pairs, remaining >= 0.
    # We want high heuristic for small remaining (tight fit).
    # Use 1 / (remaining + epsilon)
    eps = 1e-9
    tightness = np.where(fits, 1.0 / (remaining + eps), 1e-9)
    np.fill_diagonal(tightness, 0)
    
    # 3. Size Prior (prefer large items? or small?)
    # Usually large items are harder to place, so placing them early is good.
    # But the heuristic is for "placing j after i".
    # Let's just use a combination of the above.
    
    candidates = []
    
    # Candidate 1: FFD only
    candidates.append(ffd_prior)
    
    # Candidate 2: Tightness only
    candidates.append(tightness)
    
    # Candidate 3: FFD * Tightness
    # This rewards pairs that are in the same FFD bin AND are tight.
    candidates.append(ffd_prior * tightness)
    
    # Candidate 4: FFD + Tightness
    candidates.append(ffd_prior + tightness)
    
    # Candidate 5: FFD^2 * Tightness (Emphasize FFD structure)
    candidates.append((ffd_prior ** 2) * tightness)
    
    # Candidate 6: FFD * (1 + Tightness)
    candidates.append(ffd_prior * (1 + tightness))
    
    # Normalize candidates to have similar scale?
    # The ACO uses pheromone * heuristic. Pheromone starts at 1.
    # If heuristic values are very large, they dominate initially.
    # If very small, pheromone dominates.
    # Let's normalize each candidate to have max value 1.0 for fair comparison in selection.
    normalized_candidates = []
    for h in candidates:
        h_max = np.max(h)
        if h_max > 0:
            h_norm = h / h_max
        else:
            h_norm = h
        # Ensure non-finite values are handled
        h_norm = np.where(np.isfinite(h_norm), h_norm, 1e-9)
        normalized_candidates.append(h_norm)
        
    # Evaluate candidates
    best_heuristic = normalized_candidates[0]
    best_score = float('inf')
    
    for idx, h in enumerate(normalized_candidates):
        if time.time() > deadline - 60:
            break
        score = _fast_aco_eval(demands, capacity, h, n_ants=5, n_iters=5, seed=42)
        if score < best_score:
            best_score = score
            best_heuristic = h
            
    # Final check: ensure the returned matrix is valid (finite, correct shape)
    if not np.all(np.isfinite(best_heuristic)):
        best_heuristic = np.where(np.isfinite(best_heuristic), best_heuristic, 1e-9)
        
    return best_heuristic
