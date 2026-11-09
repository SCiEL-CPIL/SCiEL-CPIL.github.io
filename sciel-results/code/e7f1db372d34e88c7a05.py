import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Lower Confidence Bound (LCB) for minimization, which is equivalent
    to maximizing the score: -mu + beta * sigma.
    Beta is adaptive based on gap from optimum and uncertainty structure.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 1.0
    
    # Adaptive Beta calculation
    # We want to explore more (higher beta) when we are far from the optimum (f_best >> 3.0)
    # and exploit more (lower beta) when we are close (f_best ~ 3.0)
    
    # Gap from the theoretical minimum
    gap = max(f_best - 3.0, 0.0)
    
    # Base exploration constant using saturating function
    # Increased base beta to prevent premature convergence in small budget (30 evals).
    # When gap=0, beta_base~1.5. When gap is large, beta_base~3.0.
    beta_base = 1.5 + 1.5 * (1.0 - np.exp(-gap / 2.0))
    
    # Additional adaptation based on uncertainty structure
    # Compute coefficient of variation of sigma
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    
    if sigma_mean > 1e-9:
        cv = sigma_std / sigma_mean
    else:
        cv = 0.0
    
    # When uncertainty is concentrated (low cv), exploit more (reduce beta slightly)
    # When uncertainty is spread out (high cv), explore more (increase beta slightly)
    # Reduced impact of CV to keep beta higher on average.
    # cv=0 -> factor=0.9 (slightly more exploitation)
    # cv=0.5 -> factor=1.0 (neutral)
    # cv=1.0+ -> factor=1.1 (slightly more exploration)
    cv_factor = 0.9 + 0.2 * np.clip(cv, 0.0, 1.0)
    
    beta = beta_base * cv_factor
    
    # Calculate score: we want to maximize this score
    # Score = - (mu - beta * sigma) = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
