"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    
    Strategy:
    1. Use Expected Improvement (EI) instead of LCB. EI provides a more robust
       measure of the potential value of an evaluation by integrating over the
       posterior uncertainty. It naturally balances exploration (high sigma)
       and exploitation (low mu) without needing a manually tuned beta parameter
       that can be fragile.
    2. Incorporate cost-awareness by dividing the EI by the cost of evaluation.
       This encourages the algorithm to prioritize evaluations that are both
       promising and cheap.
    3. Implement a dynamic exploration strategy. Early in the optimization,
       we might want to explore more. However, EI inherently adapts: when f_best
       is far from mu, EI is small. When sigma is high, EI is large.
       
       To further refine, we can scale the improvement by a factor that accounts
       for the remaining budget or the current state. A simple and effective
       approach is to use a temperature parameter or simply rely on the cost
       normalization which effectively penalizes expensive exploratory moves.
       
       However, a key insight for cost-aware BO is that we should maximize
       EI(x) / cost(x). This is the expected improvement per unit cost.
       
       We also need to handle the case where f_best is very low (close to the
       true minimum) and sigma is small. In this case, EI will be very small
       everywhere, leading to arbitrary selections. This is fine, as the
       optimization is nearly complete.
       
       To avoid numerical issues and ensure stability:
       - Clip sigma to a small positive value.
       - Use a stable implementation of the normal CDF and PDF.
       - Handle the case where f_best is None or invalid (though the contract
         implies it's the lowest observed value).
       
       The cost is given as `cost`. The problem states cost = 1 + mean(x scaled to [0,1]).
       Since we don't have x, we just use the provided `cost` vector.
       
       EI formula for minimization:
       Let Z = (f_best - mu) / sigma
       EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
       where Phi is the standard normal CDF and phi is the standard normal PDF.
       
       We maximize EI / cost.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin of 10s before 300s limit
    
    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Safety check for finite inputs
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return np.zeros(n, dtype=np.float64)
    
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Ensure sigma is non-negative (it's a standard deviation)
    # Add a small epsilon to avoid division by zero in Z calculation
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Calculate Z = (f_best - mu) / sigma
    # We want to minimize f, so improvement is f_best - f(x).
    # If mu is much lower than f_best, Z is large positive.
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # To avoid overflow/underflow in exp, we can use stable forms, but for
    # typical BO ranges, standard scipy functions are usually fine.
    # However, for very large z, Phi(z) -> 1, phi(z) -> 0.
    # For very negative z, Phi(z) -> 0, phi(z) -> 0.
    
    # Using scipy.stats.norm
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # EI should be non-negative. Due to numerical errors, it might be slightly negative.
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    scores = ei / cost_safe
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.time() > deadline:
        # If we somehow took too long, return what we have (which is already computed)
        pass
        
    return scores
# EVOLVE-BLOCK-END
