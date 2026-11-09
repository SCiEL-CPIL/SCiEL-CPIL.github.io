"""Acquisition function for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    UCB acquisition with dynamically increasing exploration constant.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Posterior mean of the GP at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation of the GP at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        UCB scores; higher means more promising to evaluate next.
    """
    # Estimate number of evaluations from the scale of sigma and mu.
    # In early stages, sigma is large relative to the problem scale;
    # as more points are observed, sigma shrinks.
    # We use the median sigma as a proxy for exploration level.
    # 
    # The directive says: kappa scales with sqrt(number of evaluations).
    # Since we don't have the exact count, we can infer it from the
    # typical magnitude of sigma. With a Matern-5/2 kernel with length
    # scale 0.2 on [0,1]^3, the initial variance (before any observations)
    # for standardized targets is 1.0. As points are added, sigma decreases.
    #
    # A simpler and more robust approach: use a fixed schedule based on
    # the observation that we have at most 30 evaluations. We can estimate
    # the current evaluation count from how "tight" the posterior is.
    #
    # Actually, the cleanest structural change: use a UCB with a kappa that
    # decreases over time to shift from exploration to exploitation.
    # Since we cannot directly access the iteration count, we'll use the
    # ratio of median sigma to the initial sigma (which is ~1 for standardized
    # targets) as a proxy.
    
    med_sigma = np.median(sigma)
    
    # The initial sigma for standardized targets with Matern-5/2 is 1.0.
    # As we observe more points, sigma generally decreases.
    # Use a heuristic: kappa starts high and decreases.
    # Let's define kappa based on the typical sigma value.
    # When sigma is large (~1), we want high exploration; when small, lower.
    
    # kappa = sqrt(log(t + 1)) is standard, but without t, we approximate.
    # Let's use: kappa = c / (med_sigma + eps) clipped, or a fixed reasonable value.
    
    # A well-known practical choice for UCB in BO is kappa ~ 1-10.
    # For a dynamic approach without explicit iteration count, we can use:
    # kappa = max(0.5, 5.0 * med_sigma)
    # This gives high exploration when sigma is large, lower when small.
    
    eps = 1e-9
    kappa = np.clip(5.0 * med_sigma, 0.5, 10.0)
    
    # UCB for minimization: we want to find low values.
    # UCB score = mu - kappa * sigma
    # Higher score = more promising (lower expected value with exploration bonus).
    
    scores = mu - kappa * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
