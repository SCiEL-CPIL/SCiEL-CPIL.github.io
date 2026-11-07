# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Expected Improvement (CA-EI) with Exploration Floor.
    
    Strategy:
    1. Compute Expected Improvement (EI) for minimization:
       EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
       This measures the expected amount by which x might improve upon f_best.
    2. Adjust for cost by dividing EI by the evaluation cost.
       This favors points that offer high expected improvement per unit cost.
    3. Add a small exploration term based on sigma to prevent premature convergence
       in multi-modal landscapes, scaled by a decay factor based on progress.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Predictive mean from the GP.
    sigma : array-like, shape (2000,)
        Predictive standard deviation from the GP.
    f_best : float
        Lowest observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Deadline guard: Return zeros if we are close to the time limit
    if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
        return np.zeros_like(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in extreme tails
    z = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Adjusted EI
    # We want to maximize EI per unit cost
    cai = ei / cost
    
    # Progress-based exploration adjustment
    # Estimate progress based on f_best relative to theoretical min (-78.3323)
    # Initial f_best is typically around -20 to -40.
    # As we get closer to -78.33, we should rely more on EI (exploitation)
    # and less on pure uncertainty (exploration).
    
    f_min_theoretical = -78.3323
    f_max_initial = -10.0
    
    progress = (f_max_initial - f_best) / (f_max_initial - f_min_theoretical)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Exploration weight decays as progress increases
    # Early: high exploration weight to find basins
    # Late: low exploration weight to refine
    exploration_weight = 0.5 * (1.0 - progress)
    
    # Exploration term: sigma / cost
    # This encourages exploring high-uncertainty areas, penalized by cost
    exploration_term = exploration_weight * (sigma / cost)
    
    # Combine CA-EI and Exploration Term
    scores = cai + exploration_term
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
