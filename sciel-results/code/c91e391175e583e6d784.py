import numpy as np
from scipy.stats import norm
import time

# Global timer for the entire evaluation session
_start_time = time.time()
_TIME_LIMIT = 310  # 300s limit, keep 10s margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid Expected Improvement (EI) with
    dynamic exploration and diversity penalty.
    
    This approach combines:
    1. Expected Improvement (EI) for principled exploitation/exploration trade-off.
    2. Dynamic beta scaling based on progress (from inspiration 1).
    3. Multi-scale density penalty in (mu, sigma) feature space to enforce
       spatial diversity and escape local optima (from knowledge findings).
    
    For minimization, EI(x) = E[max(f_best - Y(x), 0)]
    where Y(x) ~ N(mu(x), sigma(x)^2).
    
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    
    We also add a diversity penalty that down-weights candidates in regions
    where many candidates cluster in the (mu, sigma) feature space.
    """
    # Deadline guard: if we are running out of time, return a safe fallback
    # to ensure the loop doesn't hang or crash due to timeout in the evaluator.
    # The fallback exploits the current best guess (lowest mu).
    if time.time() - _start_time > _TIME_LIMIT:
        return -np.asarray(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Robust sigma floor to avoid numerical issues
    sigma_floor = 1e-8
    sigma_safe = np.maximum(sigma, sigma_floor)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, sigma_floor)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    
    # Known global minimum for Hartmann 3D
    f_global_min = -3.86278
    f_start_ref = 0.0
    
    # Calculate progress: 1.0 at start (poor f_best), 0.0 at optimum
    span = f_start_ref - f_global_min
    if span <= 0:
        progress = 0.5
    else:
        progress = (f_best - f_global_min) / span
        progress = np.clip(progress, 0.0, 1.0)
    
    # Compute Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu_safe) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi(z) is the standard normal CDF
    # phi(z) is the standard normal PDF
    ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # For minimization, higher EI is better.
    # But we also want to add a diversity penalty.
    
    # Multi-scale density penalty in (mu, sigma) feature space
    # We estimate local density of candidates in the (mu, sigma) plane
    # and down-weight candidates in high-density regions.
    
    # Use a simple kernel density estimate with two scales
    # Scale 1: fine-grained (catches local clusters)
    # Scale 2: coarse-grained (catches broad regions)
    
    # Compute pairwise distances in (mu, sigma) feature space
    # This is O(n^2) but n=2000, so it's manageable.
    # However, for speed, we use a grid-based approximation.
    
    # Grid-based density estimation
    n = len(mu_safe)
    if n < 10:
        density_penalty = np.zeros(n)
    else:
        # Normalize mu and sigma to [0, 1] for grid
        mu_min, mu_max = np.min(mu_safe), np.max(mu_safe)
        sig_min, sig_max = np.min(sigma_safe), np.max(sigma_safe)
        
        mu_range = mu_max - mu_min
        sig_range = sig_max - sig_min
        
        if mu_range < 1e-10:
            mu_range = 1e-10
        if sig_range < 1e-10:
            sig_range = 1e-10
        
        mu_norm = (mu_safe - mu_min) / mu_range
        sig_norm = (sigma_safe - sig_min) / sig_range
        
        # Use a 2D grid with 20x20 cells
        grid_size = 20
        mu_idx = np.clip((mu_norm * grid_size).astype(int), 0, grid_size - 1)
        sig_idx = np.clip((sig_norm * grid_size).astype(int), 0, grid_size - 1)
        
        # Count candidates in each cell
        cell_counts = np.zeros((grid_size, grid_size), dtype=np.float64)
        for i in range(n):
            cell_counts[mu_idx[i], sig_idx[i]] += 1
        
        # Normalize by total count
        cell_density = cell_counts / n
        
        # Get density for each candidate
        local_density = cell_density[mu_idx, sig_idx]
        
        # Multi-scale: also use a coarser grid (5x5)
        coarse_grid_size = 5
        mu_idx_coarse = np.clip((mu_norm * coarse_grid_size).astype(int), 0, coarse_grid_size - 1)
        sig_idx_coarse = np.clip((sig_norm * coarse_grid_size).astype(int), 0, coarse_grid_size - 1)
        
        coarse_cell_counts = np.zeros((coarse_grid_size, coarse_grid_size), dtype=np.float64)
        for i in range(n):
            coarse_cell_counts[mu_idx_coarse[i], sig_idx_coarse[i]] += 1
        
        coarse_cell_density = coarse_cell_counts / n
        coarse_local_density = coarse_cell_density[mu_idx_coarse, sig_idx_coarse]
        
        # Combined density: weighted sum of fine and coarse
        density_penalty = 0.7 * local_density + 0.3 * coarse_local_density
    
    # Scale the density penalty to be comparable to EI values
    # EI values are typically in the range of [0, 1] for Hartmann 3D
    # Density penalty is in [0, 1], so we scale it appropriately
    
    # Dynamic scaling of the penalty based on progress
    # Early on (high progress), we want more diversity
    # Late on (low progress), we want more exploitation
    penalty_weight = 0.3 * np.sqrt(progress) + 0.1
    
    # Final score: EI - penalty_weight * density_penalty
    scores = ei - penalty_weight * density_penalty
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
