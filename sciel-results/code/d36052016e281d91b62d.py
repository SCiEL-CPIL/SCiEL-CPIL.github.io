import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    with an adaptive exploration coefficient. EI handles exploitation well,
    while UCB with adaptive beta encourages exploration in early stages and
    focuses on exploitation as the GP converges.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive beta for UCB:
    # Use the standard deviation of predicted means as a proxy for global uncertainty.
    # When mu has high variance (early in optimization), we explore more.
    # As the GP converges and mu variance decreases, we exploit more.
    mu_std = np.std(mu)
    
    # Scale beta between ~1.0 and ~3.5 based on mu_std
    # Using a smooth saturation function
    beta = 1.0 + 2.5 * (mu_std / (mu_std + 0.15))
    
    # UCB score for minimization: maximize -mu + beta * sigma
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize both EI and UCB scores to [0, 1] range for combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    if ei_max - ei_min > 1e-10:
        ei_normalized = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_normalized = np.zeros_like(ei)
    
    ucb_min, ucb_max = np.min(ucb_score), np.max(ucb_score)
    if ucb_max - ucb_min > 1e-10:
        ucb_normalized = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_normalized = np.zeros_like(ucb_score)
    
    # Weighted combination: 60% exploitation (EI), 40% exploration (UCB)
    # This balances finding good solutions with exploring uncertain regions
    w_ei = 0.6
    w_ucb = 0.4
    scores = w_ei * ei_normalized + w_ucb * ucb_normalized
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
