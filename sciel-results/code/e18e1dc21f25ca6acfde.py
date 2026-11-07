import numpy as np
import time

def _simulate_aco_fast(demands, capacity, H, n_ants=20, n_iterations=30, seed=42):
    """
    Simulate the ACO process to evaluate the heuristic matrix H.
    Optimized for speed using pre-computed structures where possible.
    """
    W = np.array(demands, dtype=float)
    n = len(W)
    
    if n == 0:
        return 0.0

    rng = np.random.default_rng(seed)
    
    # Pheromone initialization
    P = np.ones((n, n))
    
    best_bins_per_iter = []
    
    for it in range(n_iterations):
        min_bins_this_iter = float('inf')
        
        for ant in range(n_ants):
            bins = []
            current_bin_load = 0.0
            current_item = -1
            used = np.zeros(n, dtype=bool)
            
            # Place first item randomly
            first_idx = rng.integers(n)
            current_bin_load = W[first_idx]
            current_item = first_idx
            used[first_idx] = True
            
            # Repeatedly pick fitting item
            while True:
                # Find fitting items: W[j] <= capacity - current_bin_load
                remaining_cap = capacity - current_bin_load
                fits_mask = (W <= remaining_cap) & (~used)
                
                if not np.any(fits_mask):
                    # Open new bin
                    bins.append(current_bin_load)
                    # Reset
                    current_bin_load = 0.0
                    current_item = -1
                    # Pick a random fitting item for the new bin
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
                P_row = P[current_item, :]
                H_row = H[current_item, :]
                scores = P_row * H_row
                
                # Mask out non-fitting items
                # We can use np.where to zero out invalid items
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
        P = P * 0.9 + 1.0 / min_bins_this_iter
    
    return np.mean(best_bins_per_iter)


def _generate_heuristic_alpha_beta(W, cap, alpha, beta):
    """
    Generate heuristic: H_ij = (1 / (gap_ij + eps))^alpha * w_j^beta
    for fitting pairs, 1e-9 otherwise.
    """
    n = len(W)
    pair_sums = W[:, None] + W[None, :]
    fit_mask = pair_sums <= cap
    
    gap = cap - pair_sums
    # Avoid division by zero for fitting items (gap >= 0)
    # If gap is 0, 1/gap is inf. We add a small epsilon.
    inv_gap = np.where(fit_mask, 1.0 / (gap + 1e-6), 0.0)
    w_j = W[None, :]
    
    H = np.zeros((n, n))
    # Compute only for fitting entries to save time and avoid 0^0 issues if any
    H[fit_mask] = (inv_gap[fit_mask] ** alpha) * (w_j[fit_mask] ** beta)
    
    # Ensure positivity and finiteness
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    Uses Simulated Annealing to optimize alpha and beta parameters of the heuristic.
    """
    start_time = time.time()
    deadline = start_time + 290  # 290s limit, keep 10s margin for safety
    
    W = np.array(demands, dtype=float)
    n = len(W)
    
    if n == 0:
        return np.zeros((0, 0))
    
    # Simulated Annealing setup
    # Parameters to optimize: alpha (weight of tightness), beta (weight of size)
    current_alpha = 1.0
    current_beta = 0.0
    
    # Initial evaluation
    current_H = _generate_heuristic_alpha_beta(W, capacity, current_alpha, current_beta)
    current_score = _simulate_aco_fast(demands, capacity, current_H, seed=42)
    
    best_alpha = current_alpha
    best_beta = current_beta
    best_score = current_score
    best_H = current_H
    
    temperature = 0.5
    cooling_rate = 0.95
    min_temperature = 0.01
    
    # We will run until time runs out or temperature is too low
    # We need to be careful not to exceed the time limit during the simulation step.
    
    while temperature > min_temperature:
        if time.time() > deadline - 30:  # Stop if not enough time left for a full step + eval
            break
            
        # Generate neighbors
        # Perturb alpha and beta
        delta_alpha = np.random.normal(0, 0.2)
        delta_beta = np.random.normal(0, 0.2)
        
        new_alpha = max(0.0, current_alpha + delta_alpha)
        new_beta = max(0.0, current_beta + delta_beta)
        
        # Cap parameters to reasonable ranges to avoid numerical issues
        new_alpha = min(new_alpha, 5.0)
        new_beta = min(new_beta, 5.0)
        
        new_H = _generate_heuristic_alpha_beta(W, capacity, new_alpha, new_beta)
        
        try:
            new_score = _simulate_aco_fast(demands, capacity, new_H, seed=42)
        except Exception:
            new_score = float('inf')
            
        # Acceptance probability
        if new_score < current_score:
            # Always accept improvement
            current_alpha = new_alpha
            current_beta = new_beta
            current_score = new_score
            current_H = new_H
            
            if new_score < best_score:
                best_alpha = new_alpha
                best_beta = new_beta
                best_score = new_score
                best_H = new_H
        else:
            # Accept worse solution with probability exp(-delta / T)
            delta = new_score - current_score
            if delta > 0:
                probability = np.exp(-delta / temperature)
                if np.random.random() < probability:
                    current_alpha = new_alpha
                    current_beta = new_beta
                    current_score = new_score
                    current_H = new_H
            
        temperature *= cooling_rate
        
    # Final safety check
    best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)
    
    return best_H
