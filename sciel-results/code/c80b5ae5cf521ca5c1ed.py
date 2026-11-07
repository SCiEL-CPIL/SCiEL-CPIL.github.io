import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_TIME_LIMIT = 295.0  # Leave 5s margin before 300s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Predictive Entropy Search (PES)-inspired approach.
    
    The idea is to score each candidate by the expected reduction in the entropy
    of the global minimum. We approximate this by considering the mutual information
    between the function value at the candidate and the current best.
    
    For a GP posterior, the entropy of the minimum is reduced when we observe a value
    that is likely to be lower than the current best. We approximate this using
    the probability that the candidate value is lower than f_best, weighted by
    the expected improvement.
    
    Args:
        mu: Array of mean predictions (2000,).
        sigma: Array of standard deviation predictions (2000,).
        f_best: The best (lowest) function value observed so far.
        
    Returns:
        Array of scores (2000,). Higher scores indicate better candidates.
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-8)
        # Fallback to a simple UCB-like score
        return -mu + 1.0 * sigma_safe
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # PES-inspired scoring:
    # The entropy reduction is related to how informative the observation is.
    # For minimization, an observation at x is informative if f(x) < f_best.
    # The probability that f(x) < f_best is:
    #   P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    # where Phi is the standard normal CDF.
    
    # The expected reduction in entropy can be approximated by:
    #   score(x) = P(f(x) < f_best) * (f_best - mu(x)) / sigma(x)
    # This is related to the expected improvement but weighted by the probability
    # of improvement, which captures the uncertainty.
    
    # Actually, a more standard PES approximation uses the mutual information,
    # which for a Gaussian posterior is:
    #   MI = 0.5 * (log(1 + sigma_x^2 / sigma_min^2) - ...)
    # But a simpler and effective approximation is to use the expected improvement
    # weighted by the probability of improvement, or to use a hybrid.
    
    # Let's use a combination:
    # 1. Expected Improvement (EI) for exploitation
    # 2. A term that rewards high uncertainty (exploration)
    
    # EI for minimization:
    #   EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    # where Phi and phi are the standard normal CDF and PDF.
    
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # PES-like term: probability that this point is better than current best
    # This rewards points that are likely to improve the best, especially when uncertain
    p_improve = norm.cdf(z)
    
    # Combine EI and exploration term
    # The exploration term should be larger when sigma is large and mu is not too bad
    # A simple approach: score = EI + lambda * sigma * p_improve
    # where lambda balances exploitation and exploration
    
    # For Weierstrass with many local minima, we want strong exploration
    # Let's use a moderate lambda
    lambda_explore = 0.5
    score = ei + lambda_explore * sigma_safe * p_improve
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
