"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition blending Expected Improvement (EI) with
    probability of improvement (PI) and a small uncertainty bonus.
    
    The core signal is EI (captures expected reduction in objective value),
    augmented by PI (robust probability of beating f_best) and a mild
    exploration term for high-uncertainty regions. All are normalized by
    cost to favor cheap evaluations.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points (in objective units).
    f_best : float
        Best observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    start_time = time.time()
    deadline = 320  # safety margin from 330s limit
    
    # Ensure sigma is positive
    sigma_safe = np.clip(sigma, 1e-10, None)
    
    # Compute z-score: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (probability that f(x) < f_best)
    p_improve = np.clip(Phi_z, 1e-10, 1.0 - 1e-10)
    
    # Cost normalization: use inverse cost to strongly favor cheap points
    # This aligns with the high-scoring "quantile-based" (PI/cost) approach
    cost_denom = cost
    
    # Structural change: Pure Expected Improvement per unit cost (EIC)
    # This focuses on maximizing the expected reduction in objective value
    # relative to the cost, removing the potentially diluting PI term
    # and the f_best scaling which might suppress signal when f_best is large.
    
    # Cost-aware Expected Improvement
    score = ei / cost_denom
    
    # Add a small exploration bonus for high uncertainty regions
    # to prevent getting stuck in local optima
    # Scale by inverse cost to maintain cost-awareness
    exploration_bonus = sigma_safe / cost_denom
    score = score + 0.1 * exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - start_time > deadline:
        pass  # The computation is already done, just return
    
    return score
# EVOLVE-BLOCK-END
