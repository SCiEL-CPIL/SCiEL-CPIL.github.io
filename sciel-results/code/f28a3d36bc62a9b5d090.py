"""Rewritten program using EHypR (Expected Hypervolume Ratio) acquisition for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using EHypR (Expected Hypervolume Ratio) for minimization.
    
    EHypR directly optimizes the expected ratio of best observed value to new value,
    which balances probability of improvement with magnitude of potential gain.
    
    For minimization with f_best > 0:
    EHypR(x) = E[max(f_best, f(x)) / max(f(x), epsilon)]
    
    This is computed via numerical integration over the Gaussian posterior.
    """
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure f_best is positive for the ratio computation
    f_best_safe = max(f_best, 1e-6)
    
    # Compute EHypR via quadrature
    # E[max(f_best, f) / max(f, eps)] where f ~ N(mu, sigma)
    # We integrate over f:
    # For f < f_best: integrand = f_best / max(f, eps)
    # For f >= f_best: integrand = f / max(f, eps) = 1 (when f > eps)
    
    # Since f_best > 0 and we're minimizing, most mass will be in f < f_best region
    # where integrand = f_best / f (for f > 0) or f_best/eps (for f <= 0)
    
    # Use Gauss-Hermite quadrature for efficiency
    # Transform: f = mu + sigma * sqrt(2) * t
    # Integral becomes: integral over t of g(mu + sigma*sqrt(2)*t) * exp(-t^2) * sqrt(pi)
    
    # Number of quadrature points
    n_quad = 32
    t, w = np.polynomial.hermite.hermgauss(n_quad)
    
    # f values at quadrature points
    f_vals = mu[:, None] + sigma[:, None] * np.sqrt(2) * t[None, :]
    
    # Compute integrand
    # For f > 0: max(f_best, f) / f
    # For f <= 0: max(f_best, f) / eps = f_best / eps (since f_best > 0 > f)
    
    eps = 1e-8
    
    # Handle the two cases
    # f > 0: ratio = max(f_best, f) / f
    # f <= 0: ratio = f_best / eps
    
    # Avoid division by zero
    f_safe = np.where(f_vals > eps, f_vals, eps)
    
    # max(f_best, f) / f_safe
    # When f >= f_best: f / f = 1
    # When f < f_best: f_best / f
    numerator = np.maximum(f_best_safe, f_vals)
    integrand = numerator / f_safe
    
    # For f <= 0, the ratio should be f_best/eps, which is what we get since
    # numerator = f_best (since f_best > 0 >= f) and f_safe = eps
    # So this is already correct
    
    # Numerical stability: clip very large values
    integrand = np.clip(integrand, 0, 1e6)
    
    # Perform the integration
    # E[g(f)] = integral g(f) * N(f; mu, sigma) df
    # = integral g(mu + sigma*sqrt(2)*t) * exp(-t^2) * sqrt(pi) dt / sqrt(2*pi) * ... 
    # Actually, using Hermite: E[g(X)] where X~N(mu, sigma^2)
    # = (1/sqrt(2*pi)) * integral g(x) * exp(-(x-mu)^2/(2*sigma^2)) dx
    # Substitute x = mu + sigma*sqrt(2)*t:
    # = (1/sqrt(2*pi)) * integral g(mu + sigma*sqrt(2)*t) * exp(-t^2) * sigma*sqrt(2) dt
    # = sigma * sqrt(2)/(sqrt(2*pi)) * integral g(...) * exp(-t^2) dt
    # = sigma / sqrt(pi) * integral g(...) * exp(-t^2) dt
    # Using Hermite-Gauss: integral h(t)*exp(-t^2) dt ≈ sum_i w_i * h(t_i)
    # So E[g(X)] ≈ sigma / sqrt(pi) * sum_i w_i * g(mu + sigma*sqrt(2)*t_i)
    
    score = sigma / np.sqrt(np.pi) * np.sum(integrand * w[None, :], axis=1)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # EHypR is higher when we expect a better improvement
    # But for minimization, we want to maximize this
    # The score is already "higher is better"
    
    return score
# EVOLVE-BLOCK-END
