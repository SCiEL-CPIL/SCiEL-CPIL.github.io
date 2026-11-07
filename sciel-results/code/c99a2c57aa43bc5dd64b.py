import numpy as np
import time

# Module-level state for deadline guard
_start_time = time.time()
_DEADLINE_MARGIN = 10.0  # seconds before the 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Effective Improvement (CEI) acquisition function.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviation predictions.
        f_best: float, the best observed function value so far.
        cost: (2000,) array of evaluation costs for each candidate.
        
    Returns:
        (2000,) array of acquisition scores (higher is better).
    """
    # Deadline guard: if we are close to the time limit, return a safe fallback
    # to avoid killing the entire run. Since we don't know the exact state,
    # we return a simple exploitation score (low mu / low cost) as a safe bet
    # or just return zeros to stop exploration if time is critical.
    # However, the function is called frequently. The main bottleneck is the loop.
    # The acquisition itself is vectorized and fast.
    # We will add a check here just in case this function is called in a tight loop
    # that itself might be slow, but typically the GP fit is the slow part.
    # For safety, if elapsed time > 320s, return a deterministic fallback.
    if time.time() - _start_time > 320.0:
        return -mu / cost
    
    # Cost-Effective Improvement (CEI)
    # CEI(x) = E[max(f_best - f(x), 0)] / c(x)
    # Approximation:
    # Let Y = f(x) ~ N(mu, sigma^2)
    # E[max(f_best - Y, 0)] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    # Safety: avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    # Using scipy.stats for accuracy, or manual approximation if scipy is slow?
    # scipy.stats.norm is fast for arrays.
    from scipy.stats import norm
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # Expected Improvement (EI)
    # Note: if f_best - mu is very negative, EI is ~0.
    # The formula (f_best - mu) * Phi(z) + sigma * phi(z) handles this correctly.
    # However, for large negative z, Phi(z) is ~0 and phi(z) is ~0.
    # For large positive z, Phi(z) is ~1 and phi(z) is ~0. EI ~ f_best - mu.
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Cost-Effective Improvement
    # We divide by the cost.
    # cost is given as (2000,).
    # Ensure cost is positive.
    cost_safe = np.maximum(cost, 1e-8)
    
    cei = ei / cost_safe
    
    # Handle numerical issues
    cei = np.where(np.isfinite(cei), cei, 0.0)
    
    return cei
