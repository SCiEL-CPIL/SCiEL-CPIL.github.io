# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a dynamic UCB weight that scales with the ratio of remaining budget
    to current cost, shifting from exploration to exploitation as budget depletes.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1])).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive and finite
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Dynamic exploration weight (beta)
    # We use the median sigma as a proxy for "remaining budget" / "early phase"
    # High median sigma -> early -> high beta (exploration)
    # Low median sigma -> late -> low beta (exploitation)
    
    median_sigma = np.median(sigma)
    
    # We want beta to be larger when median_sigma is large.
    # Let's normalize median_sigma by the initial expected sigma scale.
    # Since targets are standardized in GP but mu/sigma are in objective units,
    # we don't know the exact scale. However, we can use a logistic or linear mapping.
    
    # A robust heuristic:
    # beta = base_beta * (median_sigma / (median_sigma + scale))
    # If median_sigma >> scale, beta -> base_beta
    # If median_sigma << scale, beta -> 0
    
    # What is a good scale?
    # If f_best is large, the objective values are large.
    # If f_best is small, the objective values are small.
    # Let's use f_best as a scale reference?
    # Or just use a fixed scale that works well for the problem?
    
    # Given the problem is Powell 4d on [-4,5]^4, the function values can vary.
    # Let's try a dynamic scale based on the spread of mu.
    
    range_mu = np.max(mu) - np.min(mu)
    if range_mu < 1e-9:
        # If mu is constant, we rely purely on sigma?
        # But if range is 0, we are in a flat region or converged.
        # Let's set beta to a moderate value to encourage exploration if sigma is high.
        beta = 1.0 if median_sigma > 1e-3 else 0.0
    else:
        # Normalize median_sigma by range_mu
        # If sigma is comparable to the range of mu, we are in a high uncertainty phase.
        rel_sigma = median_sigma / range_mu
        
        # Map rel_sigma to beta
        # We want beta to be high when rel_sigma is high.
        # Let's use a linear scaling clipped.
        # beta = 3.0 * rel_sigma
        # Clip to [0.1, 3.0] to ensure some exploration and avoid excessive exploitation.
        beta = 3.0 * rel_sigma
        beta = np.clip(beta, 0.1, 3.0)
    
    # Cost-aware score
    # We want to maximize improvement per cost.
    # Score = (ei + beta * sigma) / cost
    
    # The EI term captures exploitation (improvement over best).
    # The beta * sigma term captures exploration (uncertainty).
    # Dividing by cost ensures we prefer cheaper evaluations.
    
    scores = (ei + beta * sigma) / cost
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
