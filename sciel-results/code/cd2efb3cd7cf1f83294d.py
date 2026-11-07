import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Compute Upper Confidence Bound (UCB) score for minimization.
    
    The score is designed such that higher values indicate more promising candidates.
    For minimization, we use:
    Score(x) = -mu(x) + kappa * sigma(x)
    
    This balances exploitation (low mu) and exploration (high sigma).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, np.inf)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 0.0)
    
    # Exploration-exploitation trade-off parameter
    # 1.0 is a standard default. 2.0 is more exploratory.
    # Given the 1D nature and limited budget, 1.0 is a safe start.
    # However, to ensure a structural difference from EI (which is improvement-focused),
    # UCB is the choice.
    kappa = 1.0
    
    scores = -mu + kappa * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
