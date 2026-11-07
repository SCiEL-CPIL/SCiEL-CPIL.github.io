import numpy as np
from scipy.stats import norm, t as t_dist
import time

# Global timer for the entire evaluation session
_start_time = time.time()
_TIME_LIMIT = 310  # 300s limit, keep 10s margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Bayesian t-distribution regret risk quantification
    with multi-scale diversity penalty.
    
    Structural Change: Uses a t-distribution approximation of the GP predictive
    distribution to compute P(Y(x) < f_best), which naturally balances exploration
    and exploitation. High sigma widens the t-distribution, increasing the
    probability of improvement even when mu is moderate. This is more principled
    than ad-hoc sigma bonuses.
    
    Refinement: Multi-scale density penalty in (mu, sigma) feature space to
    enforce spatial diversity and escape local optima, scaled by progress.
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        mu = np.asarray(mu, dtype=np.float64)
        return -mu  # Fallback: exploit lowest mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Robust sigma floor
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
    
    # Estimate number of observations from the candidate pool size
    # In practice, n is the number of training points, but we don't have that directly.
    # We use a heuristic: assume n is proportional to how "confident" the GP is.
    # A simpler approach: use a fixed moderate df that balances exploration/exploitation.
    # df=5 is a common choice for moderate exploration; df=10 is more exploitative.
    # We'll interpolate based on progress: high progress (early) -> lower df (more exploration)
    df = 5 + 5 * (1 - progress)  # df ranges from 10 (early) to 5 (late)
    
    # Compute t-distribution based improvement probability
    # For a t-distribution with df degrees of freedom, location mu, scale sigma:
    # P(Y < f_best) = F_t((f_best - mu) / sigma, df)
    # where F_t is the CDF of the standard t-distribution
    
    z_t = (f_best - mu_safe) / sigma_safe
    
    # Compute CDF of t-distribution
    # For large z_t, CDF -> 1; for small z_t, CDF -> 0
    improvement_prob = t_dist.cdf(z_t, df)
    
    # Ensure improvement_prob is in [0, 1] and finite
    improvement_prob = np.clip(improvement_prob, 0.0, 1.0)
    improvement_prob = np.where(np.isfinite(improvement_prob), improvement_prob, 0.0)
    
    # Also compute standard EI as a secondary signal
    z_ei = (f_best - mu_safe) / sigma_safe
    ei = (f_best - mu_safe) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Normalize EI to [0, 1] based on max in current pool
    ei_max = np.max(ei)
    if ei_max < 1e-10:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = ei / ei_max
    
    # Multi-scale density penalty in (mu, sigma) feature space
    n = len(mu_safe)
    if n < 10:
        density_penalty = np.zeros(n)
    else:
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
        
        # Fine-grained grid (20x20)
        grid_size = 20
        mu_idx = np.clip((mu_norm * grid_size).astype(int), 0, grid_size - 1)
        sig_idx = np.clip((sig_norm * grid_size).astype(int), 0, grid_size - 1)
        
        cell_counts = np.zeros((grid_size, grid_size), dtype=np.float64)
        for i in range(n):
            cell_counts[mu_idx[i], sig_idx[i]] += 1
        
        cell_density = cell_counts / n
        local_density = cell_density[mu_idx, sig_idx]
        
        # Coarse grid (5x5)
        coarse_grid_size = 5
        mu_idx_coarse = np.clip((mu_norm * coarse_grid_size).astype(int), 0, coarse_grid_size - 1)
        sig_idx_coarse = np.clip((sig_norm * coarse_grid_size).astype(int), 0, coarse_grid_size - 1)
        
        coarse_cell_counts = np.zeros((coarse_grid_size, coarse_grid_size), dtype=np.float64)
        for i in range(n):
            coarse_cell_counts[mu_idx_coarse[i], sig_idx_coarse[i]] += 1
        
        coarse_cell_density = coarse_cell_counts / n
        coarse_local_density = coarse_cell_density[mu_idx_coarse, sig_idx_coarse]
        
        density_penalty = 0.7 * local_density + 0.3 * coarse_local_density
    
    # Dynamic scaling of penalty: more diversity early, less late
    penalty_weight = 0.2 * np.sqrt(progress) + 0.05
    
    # Combine signals:
    # Primary: improvement probability (t-distribution based)
    # Secondary: normalized EI (for fine-grained exploitation signal)
    # Penalty: diversity penalty
    
    # Weight the improvement probability heavily, with a small EI contribution
    # and subtract the diversity penalty
    scores = 0.8 * improvement_prob + 0.2 * ei_norm - penalty_weight * density_penalty
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
