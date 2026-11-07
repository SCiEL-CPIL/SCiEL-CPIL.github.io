import numpy as np

def acquisition(mu, sigma, f_best):
    n = len(mu)
    scores = np.zeros(n)
    
    if n == 0:
        return scores

    # Number of quantile bins
    k_bins = 20
    
    # Compute quantile edges
    quantiles = np.linspace(0, 1, k_bins + 1)
    bin_edges = np.quantile(mu, quantiles)
    
    # Assign bin indices
    bin_indices = np.digitize(mu, bin_edges[1:-1])
    bin_indices = np.clip(bin_indices, 0, k_bins - 1)
    
    # For each bin, find the point with the highest sigma (most uncertain)
    # This forces systematic exploration of high-uncertainty regions
    leaders = np.zeros(n, dtype=bool)
    
    for b in range(k_bins):
        mask = (bin_indices == b)
        if not np.any(mask):
            continue
        
        indices_in_bin = np.where(mask)[0]
        
        # Find the index with the maximum sigma within this bin
        sigma_vals = sigma[indices_in_bin]
        local_best_idx = np.argmax(sigma_vals)
        global_best_idx = indices_in_bin[local_best_idx]
        
        # Score based on negative sigma: lower sigma = higher score
        # But we want to maximize, so higher sigma should get higher score
        # Actually, for exploration, we want high sigma points to be preferred
        # So score = sigma (higher is better)
        best_score = sigma_vals[local_best_idx]
        
        scores[global_best_idx] = best_score
        leaders[global_best_idx] = True
    
    # Set non-leaders to a very low value to ensure bin leaders are selected
    scores[~leaders] = -1e10
    
    # Ensure finite
    scores[~np.isfinite(scores)] = -1e10
    
    return scores
