"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level cache for adaptive parameters
# We will use a dynamic beta for UCB that depends on the number of evaluations.
# Since we don't have direct access to the iteration count in the signature,
# we can infer a proxy from the cost or just use a fixed, well-tuned beta.
# However, a better strategy is to use a "knowledge gradient" style or 
# a tuned UCB. The prompt mentions UCB with fixed kappa=1 improved score.
# Let's implement UCB with a cost-aware normalization.
# UCB(x) = mu(x) + beta * sigma(x)
# Cost-aware: (f_best - (mu(x) + beta * sigma(x))) / cost(x)  <-- Wait, we are minimizing.
# For minimization, we want low values.
# EI is (f_best - mu) * Phi(z) + sigma * phi(z).
# UCB for minimization is usually: mu(x) - beta * sigma(x).
# We want to maximize the acquisition score.
# So score = (f_best - (mu(x) - beta * sigma(x))) / cost(x)
# This is equivalent to: (f_best - mu(x) + beta * sigma(x)) / cost(x)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using UCB.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions.
    sigma : array of shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Dynamic Beta Selection
    # A common heuristic for UCB in BO is beta = 2 * sqrt(log(1 + t)) or similar.
    # Without t, we can estimate uncertainty from sigma.
    # However, a fixed beta often works well if tuned.
    # The inspiration suggests UCB with kappa=1 (which is beta in our context) improved score.
    # Let's use a slightly adaptive beta based on the magnitude of sigma relative to f_best range.
    # But to keep it simple and robust, let's stick to a fixed beta that balances exploration.
    # For Hartmann 6d, the function values are around -3 to 0.
    # sigma is in objective units.
    
    # Let's try a beta that scales with the observation noise or a fixed value.
    # The inspiration said "UCB with fixed kappa=1" improved score.
    # Let's use beta = 1.0.
    
    beta = 1.0
    
    # UCB lower bound (for minimization): L(x) = mu(x) - beta * sigma(x)
    # We want to maximize the potential improvement over f_best.
    # Improvement = f_best - L(x) = f_best - mu(x) + beta * sigma(x)
    
    improvement = f_best - mu + beta * sigma
    
    # Cost-aware score
    scores = improvement / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check
    if time.time() - start_time > 2.0:
        pass # Should not happen for vectorized ops on 2000 points
        
    return scores
# EVOLVE-BLOCK-END
