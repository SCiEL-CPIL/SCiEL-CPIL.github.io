"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 290  # Keep a 10s safety margin before the 300s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Upper Confidence Bound (UCB).
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: simple exploitation (minimize mu)
        return -mu

    # 1. Estimate the current level of uncertainty around the best point.
    # We assume that the sigma at the point with the lowest mu (which is likely
    # near f_best, though not necessarily the exact f_best point if the GP is noisy
    # or if the best point was evaluated in a previous iteration) is a good proxy
    # for the "scale" of uncertainty in the promising region.
    # However, a more robust proxy for the "current exploration budget" is the
    # maximum sigma observed, or the sigma at the best predicted point.
    
    # Let's use the sigma at the point with the minimum mu as a local uncertainty measure.
    # If the GP is confident about the minimum, this sigma will be small.
    # If the GP is still exploring, this sigma might be larger.
    
    idx_min_mu = np.argmin(mu)
    local_sigma = sigma[idx_min_mu]
    
    # To prevent division by zero or extremely small values causing huge beta,
    # we set a lower bound. Also, we want beta to be a multiplier on sigma.
    # Standard UCB: score = -mu + beta * sigma
    # We want to tune beta.
    
    # Heuristic: 
    # - Early on, local_sigma might be high (e.g., > 0.1 of the range). We want high exploration.
    # - Later, local_sigma drops. We want to exploit.
    # 
    # Let's define a target exploration level.
    # A common heuristic is to make beta proportional to the inverse of the number of iterations,
    # but we don't have the iteration count directly. 
    # We can infer progress from f_best if we knew the initial f_best, but we don't.
    
    # Alternative Strategy: 
    # Use a fixed, moderately aggressive beta, but clamp it based on the global sigma range.
    # If the sigma values are very small overall, reduce beta to avoid noise-driven exploration.
    # If sigma values are large, keep beta high.
    
    # Let's look at the typical sigma.
    median_sigma = np.median(sigma)
    
    # Scale beta based on the median uncertainty.
    # If median_sigma is large, we are in an unexplored region -> high beta.
    # If median_sigma is small, we are in a well-explored region -> low beta.
    
    # Branin 2D range is roughly [0, 300]. 
    # Standardized targets mean mu/sigma are in units of the objective function's standard deviation.
    # Wait, the prompt says "standardized targets". 
    # "mu and sigma ... are in objective units." 
    # This implies the GP has been trained on standardized targets (y - mean)/std, 
    # but the output mu/sigma are mapped back to objective units?
    # "mu and sigma (a standard deviation) are in objective units."
    # Okay, so if the objective is Branin, values are ~0-300.
    # Initial noise is 1e-6.
    
    # If targets are standardized, the GP outputs are in standardized space.
    # If the prompt says they are in objective units, it means the evaluator has already
    # transformed them back, OR the GP is trained on raw values?
    # "GP: ... standardized targets; mu and sigma ... are in objective units."
    # This is slightly contradictory unless the evaluator does the inverse transform.
    # Assuming mu and sigma are indeed in the same scale as f_best (Branin values).
    
    # Let's use a dynamic beta that decays as the uncertainty decreases.
    # beta = base_beta * (current_uncertainty / reference_uncertainty)
    
    # Reference uncertainty: The initial uncertainty of the GP before seeing data is roughly
    # the prior std. For a Matern 5/2 with length scale 0.2 on [0,1]^2, 
    # the prior variance is 1 (if standardized). 
    # In objective units, the prior std would be the std of the Branin function samples.
    # Branin std is roughly 10-20?
    
    # Let's simply use the current median sigma to scale the exploration.
    # If median sigma is > 1.0 (in objective units, assuming Branin scale), explore more.
    # If median sigma is < 0.1, exploit.
    
    # Let's try a simpler, robust adaptive strategy:
    # beta = 2.0 * (median_sigma / (median_sigma + 1.0)) + 0.1
    # This ensures beta is between 0.1 and 2.1.
    # When sigma is large, beta ~ 2.1 (high exploration).
    # When sigma is small, beta ~ 0.1 (high exploitation).
    
    # Actually, for Branin, the function is multimodal. 
    # UCB with high beta early helps jump between basins.
    
    beta = 1.5 * (median_sigma / (median_sigma + 0.5)) + 0.2
    
    # Calculate scores: -mu + beta * sigma
    # Higher score means lower mu (good) and higher sigma (exploration).
    scores = -mu + beta * sigma
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
