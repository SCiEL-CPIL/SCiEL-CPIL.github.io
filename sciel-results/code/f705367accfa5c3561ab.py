import numpy as np
import time

# Global state
_call_count = 0
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    global _call_count, _start_time
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Fallback: exploit known best (negative mean)
        return -mu
    
    _call_count += 1
    t = _call_count  # 1-based iteration count
    
    # Structural change: Use a hybrid of Expected Improvement (EI) and a 
    # distance-weighted exploration term.
    # 
    # EI for minimization:
    # improvement = f_best - mu
    # z = improvement / sigma
    # EI = improvement * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal.
    #
    # We want to maximize EI.
    # But EI can be zero if sigma is very small.
    # Also, EI is 0 when mu > f_best (no improvement expected).
    #
    # Let's combine EI with a pure exploration term based on sigma,
    # but weighted by how "far" we are from the best known region.
    #
    # Actually, a simpler and more robust structural change is to use
    # a "Thompson Sampling" style or a "Knowledge Gradient" approximation.
    #
    # Let's try a different approach: 
    # Score = -mu + beta * sigma, but beta is adaptive based on the 
    # "coverage" of the domain. Since we don't have coordinates, we can't 
    # directly measure coverage.
    #
    # Alternative structural change: 
    # Use a multi-criteria score.
    # 1. Exploit: -mu (lower is better)
    # 2. Explore: sigma (higher is better)
    # 3. Novelty: Since we don't have coords, we can't do this directly.
    #
    # Let's stick to a refined EI which is theoretically optimal for 
    # Bayesian optimization under certain assumptions.
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Handle sigma = 0 case
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # z-score
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    # Using scipy.stats.norm if available, else manual approximation
    try:
        from scipy.stats import norm
        phi = norm.pdf(z)
        Phi = norm.cdf(z)
    except ImportError:
        # Manual approximation
        # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        Phi = 0.5 * (1 + np.erf(z / np.sqrt(2)))
        # phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
        phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: if mu > f_best, z < 0, Phi(z) < 0.5, but (f_best - mu) < 0, 
    # so the first term is positive. The second term is always positive.
    # So EI is always positive? Let's check.
    # If mu >> f_best, z is large negative, Phi(z) ~ 0, phi(z) ~ 0, EI ~ 0.
    # If mu << f_best, z is large positive, Phi(z) ~ 1, phi(z) ~ 0, 
    # EI ~ f_best - mu (large positive).
    # So EI is a good measure of expected improvement.
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # To avoid pure exploitation, we can add a small exploration bonus.
    # But EI already has an exploration component via sigma * phi(z).
    # When sigma is large and z is moderate, EI is significant.
    
    # However, EI can be dominated by points with very low mu (exploit) 
    # even if sigma is small. This is usually good.
    # But if we've converged, EI becomes very small everywhere.
    
    # Let's normalize EI to be comparable across iterations.
    # Or just use it directly.
    
    # One issue: if f_best is very low, and mu is always > f_best, 
    # EI will be small. We might want to force some exploration.
    
    # Hybrid: Score = EI + gamma * sigma
    # gamma decays over time.
    
    # For t=1, gamma=0.5
    # For t=25, gamma=0.1
    gamma = 0.1 + 0.4 * np.exp(-t / 5.0)
    
    scores = EI + gamma * sigma_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
