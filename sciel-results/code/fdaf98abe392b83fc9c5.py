import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement of the Minimum (EI_min) acquisition function.
    
    This is a Knowledge-Greedy approach: for each candidate point x with
    posterior N(mu(x), sigma(x)^2), we compute the expected reduction in
    the best observed value if we evaluate at x:
    
        EI_min(x) = E[max(0, f_best - f(x))]
    
    For a univariate normal, this has a closed form:
        EI = (f_best - mu) * Phi((f_best - mu)/sigma) 
             + sigma * phi((f_best - mu)/sigma)
    
    where Phi is the standard normal CDF and phi is the PDF.
    
    This is structurally different from UCB: it directly optimizes the
    expected improvement in the objective, which naturally adapts the
    exploration-exploitation tradeoff based on the posterior uncertainty
    and the gap to the current best.
    """
    start_time = time.time()
    timeout = 320.0  # safety margin before 330s kill

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    n = len(mu)
    
    # Ensure sigma is positive for computations
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement of the minimum:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    delta = f_best - mu  # positive when f(x) is expected to be below f_best
    z = delta / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Phi and phi
    Phi = norm.cdf(z_clipped)
    phi = norm.pdf(z_clipped)
    
    # EI = delta * Phi + sigma * phi
    # Only positive when delta > 0 (i.e., f_best > mu)
    EI = delta * Phi + sigma_safe * phi
    
    # EI is always non-negative; for points where mu >> f_best, EI ~ 0
    # For points where mu << f_best and sigma small, EI ~ f_best - mu
    # For points with high sigma, EI captures the exploration value
    
    # Use EI directly as the score: higher EI means more expected improvement
    scores = EI
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time
    if time.time() - start_time > timeout:
        return scores
    
    return scores
