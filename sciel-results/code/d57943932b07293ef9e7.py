import numpy as np
import time

def _ffd_assignment(demands, capacity):
    """
    Perform First-Fit Decreasing bin packing.
    Returns an array of bin indices for each item.
    """
    n = len(demands)
    sorted_indices = np.argsort(-demands)
    
    bins = []
    bin_loads = []
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

def _build_complementarity_prior(demands, capacity):
    """
    Vectorized complementarity prior.
    Rewards pairs where the remaining capacity is close to a multiple of the minimum item size.
    """
    n = len(demands)
    min_size = np.min(demands)
    
    pair_sum = demands[:, None] + demands[None, :]
    fits = (pair_sum <= capacity)
    remaining = capacity - pair_sum
    
    # Distance to nearest multiple of min_size
    # multiples = round(remaining / min_size) * min_size
    # dist = abs(remaining - multiples)
    # Heuristic: exp(-dist / scale)
    
    multiples = np.round(remaining / min_size) * min_size
    dist = np.abs(remaining - multiples)
    
    # Scale for exponential decay
    scale = 0.5 * min_size
    
    # Only valid for fitting pairs
    comp = np.exp(-dist / scale)
    comp = np.where(fits, comp, 1e-9)
    
    np.fill_diagonal(comp, 0)
    
    return comp

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
    deadline = 325.0 # 5s safety margin
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # 1. FFD Prior
    item_bin = _ffd_assignment(demands, capacity)
    ffd_prior = (item_bin[:, None] == item_bin[None, :]).astype(float)
    np.fill_diagonal(ffd_prior, 0)
    
    # 2. Tightness Prior Base (slack = capacity - sum)
    pair_sum = demands[:, None] + demands[None, :]
    fits = (pair_sum <= capacity)
    remaining = capacity - pair_sum
    # Base tightness is 1/(remaining + eps). We will apply alpha exponent later.
    # To avoid log issues or 0 division, we use a safe base
    base_tightness = np.where(fits, 1.0 / (remaining + 1e-9), 1e-9)
    np.fill_diagonal(base_tightness, 0)
    
    # 3. Complementarity Prior
    comp_prior = _build_complementarity_prior(demands, capacity)
    
    # Normalize priors to [0, 1] range for stable combination
    def normalize(h):
        h_max = np.max(h)
        if h_max > 0 and np.isfinite(h_max):
            h_norm = h / h_max
        else:
            h_norm = np.zeros_like(h)
        h_norm = np.where(np.isfinite(h_norm), h_norm, 0)
        return h_norm

    ffd_norm = normalize(ffd_prior)
    comp_norm = normalize(comp_prior)
    
    # Search Space:
    # alpha: exponent for tightness (1/slack^alpha)
    # beta: weight for complementarity
    # gamma: weight for FFD
    
    candidates = []
    
    # Phase 1: Coarse Grid
    alphas_coarse = [0.5, 1.0, 2.0, 3.0, 5.0]
    betas_coarse = [0.0, 0.5, 1.0, 2.0]
    gammas_coarse = [0.0, 0.5, 1.0]
    
    best_score = float('inf')
    best_params = None
    best_heuristic = None
    
    # Pre-compute powers for tightness to speed up evaluation? 
    # tightness_alpha = base_tightness ** alpha
    # Since base_tightness values are < 1, higher alpha makes it smaller (more penalizing for loose).
    
    for alpha in alphas_coarse:
        if time.time() > deadline - 60: break
        
        # Compute tightness component for this alpha
        # Note: base_tightness is 1/(slack). So (1/slack)^alpha = base_tightness^alpha
        tight_alpha = np.power(base_tightness, alpha)
        tight_alpha = np.where(np.isfinite(tight_alpha) & (tight_alpha > 0), tight_alpha, 1e-9)
        tight_alpha_norm = normalize(tight_alpha)
        
        for beta in betas_coarse:
            if time.time() > deadline - 60: break
            
            # Comp component is fixed, just weight it
            # Heuristic = gamma * FFD + beta * Comp + (1 - gamma - beta) * Tight? 
            # Or multiplicative? 
            # Let's try additive with normalization to ensure all parts contribute.
            # H = gamma * FFD + beta * Comp + (1 - gamma - beta) * Tight
            # But weights must be positive. 
            # Let's use a weighted sum where we normalize the final result.
            
            for gamma in gammas_coarse:
                if time.time() > deadline - 60: break
                
                # Construct heuristic
                # We want to balance the three. 
                # If gamma+beta > 1, the tightness weight becomes negative, which is bad.
                # Let's normalize the weights so they sum to 1.
                w_gamma = gamma
                w_beta = beta
                w_tight = 1.0 # Base weight for tightness, scaled by (1 - gamma - beta) if positive?
                
                # Simpler: H = w_f * FFD + w_c * Comp + w_t * Tight
                # Let's fix w_t = 1.0 and vary w_f, w_c.
                # Then normalize H.
                
                h = w_gamma * ffd_norm + w_beta * comp_norm + 1.0 * tight_alpha_norm
                
                # Ensure positive
                h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
                h = normalize(h)
                
                # Evaluate
                scores = []
                # Use 3 seeds for quick robust check
                for seed in [42, 123, 999]:
                    score = _fast_aco_eval(demands, capacity, h, n_ants=5, n_iters=5, seed=seed)
                    scores.append(score)
                
                avg_score = np.mean(scores)
                
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = (alpha, beta, gamma)
                    best_heuristic = h
                    
    # Phase 2: Fine Grid around best
    if time.time() < deadline - 30 and best_params is not None:
        a_best, b_best, g_best = best_params
        
        # Refine alpha
        alpha_span = max(0.5, 0.2 * a_best)
        fine_alphas = np.linspace(max(0.1, a_best - alpha_span), a_best + alpha_span, 5)
        
        # Refine beta
        beta_span = 0.5
        fine_betas = np.linspace(max(0.0, b_best - beta_span), b_best + beta_span, 5)
        
        # Refine gamma
        gamma_span = 0.2
        fine_gammas = np.linspace(max(0.0, g_best - gamma_span), g_best + gamma_span, 5)
        
        for alpha in fine_alphas:
            if time.time() > deadline - 10: break
            
            tight_alpha = np.power(base_tightness, alpha)
            tight_alpha = np.where(np.isfinite(tight_alpha) & (tight_alpha > 0), tight_alpha, 1e-9)
            tight_alpha_norm = normalize(tight_alpha)
            
            for beta in fine_betas:
                if time.time() > deadline - 10: break
                
                for gamma in fine_gammas:
                    if time.time() > deadline - 10: break
                    
                    h = gamma * ffd_norm + beta * comp_norm + 1.0 * tight_alpha_norm
                    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
                    h = normalize(h)
                    
                    # Evaluate with 2 seeds for speed
                    scores = []
                    for seed in [42, 123]:
                        score = _fast_aco_eval(demands, capacity, h, n_ants=5, n_iters=5, seed=seed)
                        scores.append(score)
                    
                    avg_score = np.mean(scores)
                    
                    if avg_score < best_score:
                        best_score = avg_score
                        best_params = (alpha, beta, gamma)
                        best_heuristic = h

    # Fallback if no heuristic was found (should not happen)
    if best_heuristic is None:
        best_heuristic = normalize(base_tightness)

    # Final safety check
    if not np.all(np.isfinite(best_heuristic)):
        best_heuristic = np.where(np.isfinite(best_heuristic), best_heuristic, 1e-9)
        
    return best_heuristic
