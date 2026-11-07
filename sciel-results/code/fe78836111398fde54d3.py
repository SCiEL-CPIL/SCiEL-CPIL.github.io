import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Quantile (LCB) strategy.
    
    Instead of Expected Improvement (EI), which integrates the improvement,
    this function targets the lower tail of the posterior distribution.
    Specifically, it maximizes the negative of the 10th percentile of the
    posterior predictive distribution N(mu, sigma^2).
    
    score(x) = - (mu(x) + sigma(x) * Phi^{-1}(0.1))
             = -mu(x) - sigma(x) * Phi^{-1}(0.1)
    
    Since Phi^{-1}(0.1) is negative, this is equivalent to:
    score(x) = -mu(x) + c * sigma(x)
    where c = -Phi^{-1}(0.1) ≈ 1.28.
    
    This linear combination balances exploitation (low mu) and exploration (high sigma).
    """
    # Calculate the constant for the 10th percentile
    # norm.ppf(0.1) is approximately -1.28155
    c = -norm.ppf(0.1)
    
    # Compute scores: -mu + c * sigma
    # This ensures that points with low mean and/or high uncertainty get high scores.
    scores = -mu + c * sigma
    
    # Ensure numerical stability and finite outputs
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
