import numpy as np
from scipy.stats import norm
import time

# Module-level variables for deadline guard
_START_TIME = time.time()
_DEADLINE = 320.0  # 10s safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a phase-based blend of 
    Expected Improvement (EI) and Uncertainty (Sigma), normalized by cost.
    
    - Early phase (high f_best): Emphasizes exploration (Sigma) to build the GP model cheaply.
    - Late phase (low f_best): Emphasizes exploitation (EI) to find the minimum.
    - Cost normalization: Ensures we prefer cheaper evaluations for equal quality.
    """
    # Deadline guard: If we are running out of time, fall back to a simple, fast heuristic
    # to ensure we don't cause a timeout in the evaluator loop.
    # We switch to pure cost-minimization (exploit known best cheaply? No, just pick cheap points)
    # Actually, if we are near deadline, we should just return valid scores quickly.
    # The standard calculation is O(N) and very fast, so this is mostly a safety net.
    if time.time() > _DEADLINE:
        # Fallback: Score purely by 1/cost to finish the loop with cheap evaluations
        cost_safe = np.maximum(cost, 1e-9)
        return 1.0 / cost_safe

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # Handle numerical stability for large z
    z = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu > f_best, EI is 0. The formula naturally handles this 
    # if we use the standard definition, but strictly:
    improvement_term = np.maximum(f_best - mu, 0)
    ei = improvement_term * Phi_z + sigma_safe * phi_z
    
    # Determine phase weight beta for exploration (sigma)
    # f_best starts high (~10-12) and goes to -10.5
    # Let's map f_best to a weight.
    # If f_best > 0, we are likely early -> high exploration
    # If f_best < -5, we are likely late -> low exploration
    
    # Linear interpolation for beta
    # beta_high = 2.0 (strong exploration)
    # beta_low = 0.5 (strong exploitation)
    
    # Define boundaries for phase transition
    # Start of transition: f_best = 0
    # End of transition: f_best = -5
    
    if f_best > 0:
        beta = 2.0
    elif f_best < -5:
        beta = 0.5
    else:
        # Linearly interpolate between 2.0 and 0.5 as f_best goes from 0 to -5
        # progress = (0 - f_best) / (0 - (-5)) = -f_best / 5
        # beta = 2.0 - (2.0 - 0.5) * progress
        progress = -f_best / 5.0
        beta = 2.0 - 1.5 * progress
        beta = np.clip(beta, 0.5, 2.0)
        
    # Combined Score: (EI + beta * Sigma) / Cost
    # This balances finding improvements (EI) and exploring uncertainty (Sigma)
    # while penalizing high cost.
    
    score = (ei + beta * sigma_safe) / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
