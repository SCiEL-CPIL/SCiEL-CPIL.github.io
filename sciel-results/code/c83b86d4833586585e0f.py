"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Batch-aware Look-Ahead acquisition function.
    
    Simulates the next 5 evaluations by greedily selecting top-scoring candidates
    from the pool, updating posterior variance locally to approximate information gain,
    and scoring candidates based on their expected contribution to reducing the minimum value.
    """
    n = len(mu)
    
    # Initialize working copies
    mu_work = mu.copy().astype(np.float64)
    sigma_work = sigma.copy().astype(np.float64)
    
    # Scale: 2000 points over [-5, 5], so spacing is 10/2000 = 0.005
    # Length scale is 0.2 on [0,1] which corresponds to 0.2*10 = 2.0 in original units
    # So the kernel length scale in index space is 2.0 / 0.005 = 400 indices
    
    # Window for local updates: use 1.5x the length scale in index space
    window = 600
    batch_size = 5
    
    # Compute base EI scores for initial selection
    z = (f_best - mu_work) / (sigma_work + 1e-9)
    scores = (f_best - mu_work) * norm.cdf(z) + sigma_work * norm.pdf(z)
    
    # Track which points are "used"
    used = np.zeros(n, dtype=bool)
    
    # Store selected indices for later boost calculation
    selected_indices = []
    
    # Look-ahead simulation
    for step in range(batch_size):
        # Find the best candidate among unused points
        masked_scores = np.where(used, -np.inf, scores)
        best_idx = np.argmax(masked_scores)
        
        # Mark as used
        used[best_idx] = True
        selected_indices.append(best_idx)
        
        # Update variance locally: reduce sigma for nearby points
        # Compute correlation with the selected point
        i_start = max(0, best_idx - window)
        i_end = min(n, best_idx + window + 1)
        
        indices = np.arange(i_start, i_end)
        dists = np.abs(indices - best_idx)
        # Normalize by length scale in index space (400)
        r = dists / 400.0
        # Matern-5/2 correlation
        sqrt3 = np.sqrt(3.0)
        corr = (1.0 + sqrt3 * r + r**2) * np.exp(-sqrt3 * r)
        
        # Update sigma: sigma_new = sigma * sqrt(1 - corr^2)
        # But only for unused points
        mask_unused = ~used[i_start:i_end]
        if np.any(mask_unused):
            sigma_work[i_start:i_end][mask_unused] *= np.sqrt(1.0 - corr[mask_unused]**2 + 1e-12)
        
        # Recompute scores for affected region
        z_local = (f_best - mu_work[i_start:i_end]) / (sigma_work[i_start:i_end] + 1e-9)
        scores[i_start:i_end] = (f_best - mu_work[i_start:i_end]) * norm.cdf(z_local) + sigma_work[i_start:i_end] * norm.pdf(z_local)
        
        # Also update scores for used points to -inf
        scores[i_start:i_end][used[i_start:i_end]] = -np.inf
    
    # Final scores: combine EI with a boost for being near the selected top-5
    # The idea: candidates that are close to where the look-ahead simulation
    # found good points are likely to be in promising regions
    
    # Reset to original sigma for final scoring
    sigma_orig = sigma.copy().astype(np.float64)
    z_orig = (f_best - mu) / (sigma_orig + 1e-9)
    ei_orig = (f_best - mu) * norm.cdf(z_orig) + sigma_orig * norm.pdf(z_orig)
    
    # Boost candidates near the selected top-5
    boost = np.zeros(n)
    for si in selected_indices:
        i_start = max(0, si - window)
        i_end = min(n, si + window + 1)
        indices = np.arange(i_start, i_end)
        dists = np.abs(indices - si)
        r = dists / 400.0
        sqrt3 = np.sqrt(3.0)
        corr = (1.0 + sqrt3 * r + r**2) * np.exp(-sqrt3 * r)
        # Boost proportional to correlation (how informative this point is for nearby points)
        boost[i_start:i_end] += corr * 0.15
    
    # Final score: EI + boost for being in informative regions
    final_scores = ei_orig + boost
    
    # Ensure finite values
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
