import numpy as np
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s limit

def acquisition(mu, sigma, f_best):
    # Check deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        # Return a simple UCB as fallback to stay valid
        return -mu + sigma
    
    # Probability of Improvement (PI) for minimization
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # For minimization, we want low mu and low sigma near f_best
    
    # Handle sigma = 0 cases
    safe_sigma = np.maximum(sigma, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / safe_sigma
    
    # CDF of standard normal (using scipy if available, else approximation)
    try:
        from scipy.stats import norm
        pi = norm.cdf(z)
    except ImportError:
        # Approximation using error function
        pi = 0.5 * (1 + np.vectorize(_erf_approx)(z))
    
    # Add a bias toward the center (0,0) in the scaled [0,1] space
    # The Ackley 2D minimum is at (0,0) in [-5,5]^2, which is (0.5, 0.5) in [0,1]^2
    # We don't have coordinates, so we can't directly add a distance penalty.
    # Instead, we rely on the GP mean to guide us.
    
    # Combine PI with a small exploitation term: prefer lower mu
    # Use a weighted combination
    weight_pi = 1.0
    weight_exploit = 0.1
    
    # Normalize mu to [0,1] range for the exploitation term
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    if mu_max > mu_min:
        mu_norm = (mu - mu_min) / (mu_max - mu_min)
    else:
        mu_norm = np.zeros_like(mu)
    
    # Lower mu_norm means better (lower mu), so we want to maximize (1 - mu_norm)
    score = weight_pi * pi + weight_exploit * (1 - mu_norm)
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
    
    return score

def _erf_approx(x):
    """Approximation of the error function."""
    # Abramowitz and Stegun approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    sign = 1 if x >= 0 else -1
    x_abs = abs(x)
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x_abs * x_abs)
    return sign * y
