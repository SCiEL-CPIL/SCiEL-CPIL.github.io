"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining adaptive UCB and Expected Improvement (EI).
    
    Structural Change: Shifts from pure linear UCB (current program) to a hybrid 
    probabilistic EI+UCB blend (inspiration 1/2), which is more robust in 
    multi-modal landscapes like Shekel 4d.
    
    Refinement: Incorporates the adaptive beta logic from the current program, 
    which increases exploration pressure when the GP means are tightly clustered 
    (indicating potential overconfidence or stagnation).
    
    Score = (alpha * UCB_adaptive + (1 - alpha) * EI) / cost
    where:
      UCB_adaptive = (f_best - mu) + beta_adaptive * sigma
      EI = (f_best - mu) * Phi(z) + sigma * phi(z)
      beta_adaptive = base_beta + boost * cluster_fraction
    """
    start_time = time.time()
    deadline = start_time + 290

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    f_best = float(f_best)

    # --- Adaptive Beta Calculation (Refinement from Current Program) ---
    # Detect if the GP is overconfident or stuck in a local region.
    mu_min = np.min(mu)
    mu_spread = np.std(mu)
    
    if mu_spread < 1e-6:
        mu_spread = 1e-6
        
    # Calculate fraction of points that are "close" to the best mean
    # This serves as a proxy for how much the search has converged
    cluster_ratio = (mu - mu_min) / (mu_spread + 1e-6)
    close_mask = cluster_ratio < 0.5
    close_fraction = np.mean(close_mask)
    
    # Base beta is 2.0 (standard for UCB).
    # We add a boost if a large fraction of points are clustered near the best.
    # This forces the UCB component to explore more aggressively to escape the cluster.
    base_beta = 2.0
    boost_factor = 1.5
    beta = base_beta + boost_factor * close_fraction

    # --- Component 1: Adaptive UCB ---
    ucb = (f_best - mu) + beta * sigma

    # --- Component 2: Expected Improvement (EI) ---
    # EI provides a calibrated estimate of improvement, less prone to 
    # over-exploration than UCB in high-dim spaces.
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # --- Blend UCB and EI ---
    # Alpha = 0.6 favors UCB (exploration) but includes EI (exploitation/calibration).
    # This structure (Inspiration 1) scored higher than pure UCB.
    alpha = 0.6
    combined = alpha * ucb + (1.0 - alpha) * ei

    # --- Cost Awareness ---
    scores = combined / cost

    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, -np.inf)

    if time.time() > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
