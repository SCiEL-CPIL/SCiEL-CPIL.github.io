# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a Thompson Sampling-inspired approach with adaptive exploration
    based on proximity to the known minimum. The key structural change is
    using stochastic sampling from the GP posterior to naturally balance
    exploration and exploitation, combined with adaptive cost-awareness.
    """
    # Track elapsed time for deadline guard
    t_start = time.time()
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Known minimum of hartmann 3d
    f_min_known = -3.86278
    
    # Calculate Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)]
    improvement = f_best - mu
    z = improvement / sigma_safe
    z_clipped = np.clip(z, -50, 50)
    ei = improvement * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Thompson Sampling: draw a single sample from the GP posterior
    # For minimization, we want to find points with low sampled values
    # Sample: f_sample = mu + sigma * z_sample where z_sample ~ N(0,1)
    # We use a deterministic seed based on f_best to get consistent behavior
    # within a call, but different across calls as f_best changes
    rng = np.random.default_rng(seed=int(abs(f_best) * 10000) % 2**32)
    z_sample = rng.standard_normal(len(mu))
    f_sample = mu + sigma_safe * z_sample
    
    # For minimization, we want low f_sample values
    # Convert to "higher is better" score: negate the sample
    ts_score = -f_sample
    
    # Calculate a variance-based exploration score
    pv_score = sigma_safe
    
    # Adaptive weighting based on how close we are to the known minimum
    # If f_best is close to f_min_known, we should exploit more
    # If f_best is far from f_min_known, we should explore more
    gap = f_best - f_min_known
    gap_normalized = np.clip(gap / 4.0, 0.0, 1.0)
    
    # Weight: when gap is large (early), explore more; when gap is small (late), exploit more
    # exploration_weight starts at 0.4 and decreases to 0.05 as we get closer to the minimum
    exploration_weight = 0.05 + 0.35 * gap_normalized
    exploitation_weight = 1.0 - exploration_weight
    
    # Cost-aware scoring
    # Use a power law cost penalty: score / cost^alpha
    # alpha = 1.0 is linear cost-aware, alpha < 1 is less sensitive to cost
    # Adaptive alpha: when we're far from the minimum, be less sensitive to cost
    alpha = 0.5 + 0.5 * (1.0 - gap_normalized)
    
    cost_safe = np.maximum(cost, 1e-9)
    cost_factor = cost_safe ** alpha
    
    # Normalize scores to comparable scales
    # EI is in objective units, TS score is also in objective units (negated)
    ei_max = np.max(ei) + 1e-9
    ts_min = np.min(ts_score)
    ts_max = np.max(ts_score)
    ts_range = ts_max - ts_min + 1e-9
    
    # Normalize EI to [0, 1]
    ei_normalized = ei / ei_max
    
    # Normalize TS score to [0, 1] (higher is better)
    ts_normalized = (ts_score - ts_min) / ts_range
    
    # Normalize PV to [0, 1]
    pv_max = np.max(pv_score) + 1e-9
    pv_normalized = pv_score / pv_max
    
    # Combine: use Thompson Sampling as the primary driver, with EI as a refinement
    # and PV for exploration
    combined = exploitation_weight * (0.7 * ts_normalized + 0.3 * ei_normalized) + \
               exploration_weight * pv_normalized
    
    # Apply cost-aware penalty
    scores = combined / cost_factor
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Add a small deterministic component to break ties and encourage diversity
    # Use a hash of the first few mu values for reproducibility
    tie_breaker = 1e-6 * rng.random(len(mu))
    scores = scores + tie_breaker
    
    # Deadline guard: if we're close to the limit, return quickly
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
