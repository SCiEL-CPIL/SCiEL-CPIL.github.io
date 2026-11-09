"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # 10s margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a model-and-optimize approach for EI.
    
    1. Defines a family of cost-aware EI acquisition functions parameterized by gamma.
    2. Optimizes gamma by selecting the value that maximizes the Expected Improvement (EI)
       of the best candidate identified by that EI variant.
    3. Returns the scores for the optimal gamma.
    """
    # Ensure sigma and cost are positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Define a grid of gamma values to test
    # Gamma controls the cost penalty in EI: EI(x) / (cost(x) ^ gamma)
    gammas = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    
    best_gamma = 1.0
    best_ei = -np.inf
    best_scores = None
    
    # Precompute EI for all points (independent of gamma)
    # EI(x) = (f_best - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
    # where Z = (f_best - mu(x)) / sigma(x)
    
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # Handle non-finite EI
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    for gamma in gammas:
        # Check time budget
        if time.time() > _DEADLINE - 5:
            break
            
        # Cost-aware EI score: EI(x) / (cost(x) ^ gamma)
        if gamma == 0.0:
            scores = ei
        else:
            scores = ei / (cost ** gamma)
        
        # Handle non-finite scores
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        
        # Find the best candidate according to this gamma
        idx_max = np.argmax(scores)
        
        # Calculate the EI of the selected point (this is our objective for model-and-optimize)
        # We want to maximize the EI of the point that this acquisition would select
        ei_selected = ei[idx_max]
        
        if ei_selected > best_ei:
            best_ei = ei_selected
            best_gamma = gamma
            best_scores = scores.copy()
            
    # If for some reason best_scores is None (should not happen), fallback
    if best_scores is None:
        if best_gamma == 0.0:
            best_scores = ei
        else:
            best_scores = ei / (cost ** best_gamma)
        best_scores = np.where(np.isfinite(best_scores), best_scores, 0.0)
        
    # Ensure the returned scores are finite for the final output
    # The scores used for selection were -inf for invalids, but we return 0.0 for safety
    # as per task requirement "Return 2000 finite scores"
    return np.where(np.isfinite(best_scores), best_scores, 0.0)
# EVOLVE-BLOCK-END
