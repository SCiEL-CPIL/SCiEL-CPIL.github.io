import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Lookahead Expected Improvement (L-Opt) acquisition function.
    
    This is a two-step myopic acquisition that approximates the expected
    improvement of the *next* evaluation given the current candidate is selected.
    It combines:
      1. Standard Expected Improvement (EI) for the immediate point.
      2. A lookahead term that estimates how much the posterior uncertainty
         will decrease after evaluating this point, benefiting subsequent samples.
    
    The lookahead is approximated analytically: evaluating a point with high
    uncertainty (sigma) reduces the expected posterior variance in its neighborhood,
    which increases the expected improvement of future samples. We approximate
    this by weighting EI with a factor that accounts for the "value of information"
    of reducing uncertainty.
    
    Score = EI(x) + lambda * sigma(x) * log(1 + sigma(x) / (f_best - mu(x) + eps))
    
    This encourages points where:
    - Immediate improvement is likely (high EI)
    - Uncertainty reduction will be valuable (high sigma relative to gap to best)
    """
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: clip sigma to avoid log(0) or division issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Gap from current mean to best (positive means current mean is worse than best)
    gap = f_best - mu
    
    # Standardized improvement: how many sigma below f_best the mean is
    z = gap / sigma_safe
    
    # Standard Expected Improvement for minimization:
    # EI = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(-z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # For minimization, we want f(x) < f_best, so improvement = f_best - f(x)
    # EI = (f_best - mu) * Phi(-(f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Compute EI
    phi = norm.pdf(z_clipped)
    Phi_neg = norm.cdf(-z_clipped)
    
    EI = gap * Phi_neg + sigma_safe * phi
    
    # Ensure EI is non-negative
    EI = np.maximum(EI, 0.0)
    
    # Lookahead term: value of information from reducing uncertainty
    # Approximation: sigma * log(1 + sigma / (|gap| + eps))
    # This captures the idea that high uncertainty in a region far from the best
    # has more value in reducing (could be a much better point)
    eps = 1e-9
    lookahead = sigma_safe * np.log1p(sigma_safe / (np.abs(gap) + eps))
    
    # Weight the lookahead term
    # Use a small lambda to balance immediate EI with future benefit
    lambda_lo = 0.3
    
    scores = EI + lambda_lo * lookahead
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
