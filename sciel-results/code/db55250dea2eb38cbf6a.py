"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # 10s margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a model-and-optimize framework.
    
    Strategy:
    1. Compute a base Expected Improvement (EI) score for each candidate.
    2. Define a family of cost-aware acquisition functions parameterized by a penalty exponent gamma:
       score(x) = EI(x) / (cost(x) ** gamma)
    3. To select gamma, we use a model-and-optimize approach:
       - For each candidate gamma in a grid, we determine the best candidate x* (argmax score).
       - We then evaluate the "quality" of this choice by looking at the predicted improvement 
         normalized by the cost at that specific point. Specifically, we maximize:
         J(gamma) = EI(x*(gamma)) / (cost(x*(gamma)) ** 1.0)
         This ensures that the selected gamma leads to a point that offers high improvement 
         per unit cost, rather than just high raw improvement or low cost.
    4. Return the scores corresponding to the optimal gamma.
    """
    # Ensure sigma and cost are positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Define a grid of gamma values to test.
    # Gamma controls how aggressively we penalize high-cost points.
    # A larger gamma means we prefer lower-cost points more strongly.
    gammas = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    
    best_gamma = 1.0
    best_j = -np.inf
    best_scores = None
    
    # Precompute EI for all points (independent of gamma)
    # EI(x) = (f_best - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
    # where Z = (f_best - mu(x)) / sigma(x)
    
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # Handle non-finite EI
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Precompute cost powers for efficiency
    # We need cost ** gamma for various gammas
    # And cost ** 1.0 for the objective J
    
    for gamma in gammas:
        # Check time budget
        if time.time() > _DEADLINE - 5:
            break
            
        # Compute scores for this gamma
        if gamma == 0.0:
            scores = ei.copy()
        else:
            scores = ei / (cost ** gamma)
        
        # Handle non-finite scores
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        
        # Find the best candidate according to this gamma
        idx_max = np.argmax(scores)
        
        # Calculate the objective J for this gamma:
        # J = EI(x*) / cost(x*)
        # This represents the expected improvement per unit cost at the selected point.
        ei_selected = ei[idx_max]
        cost_selected = cost[idx_max]
        
        # If cost_selected is very small, J is large. If ei_selected is 0, J is 0.
        j_val = ei_selected / cost_selected
        
        # We want to maximize J. However, we should also ensure that the selected point
        # actually has a reasonable EI. If EI is 0, J is 0.
        
        if j_val > best_j:
            best_j = j_val
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
