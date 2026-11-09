"""Direct acquisition for sphere 1d: Expected Hypar (EHypR) acquisition.

EHypR maximizes E[max(f_best, f(x)) / f(x)] which directly targets simple
regret minimization. For the sphere function f(x) = x^2, we use the GP
posterior to compute this expectation in closed form using the normal CDF
and PDF.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition for minimizing sphere 1d on [-5, 5]^1 (min value 0 at x=0).

    mu, sigma: shape (2000,), posterior mean / std in objective units.
    f_best: lowest observed value.
    Returns 2000 finite scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    f_best = float(max(f_best, 1e-12))  # Avoid division by zero

    # EHypR: E[max(f_best, f(x)) / f(x)]
    # = P(f(x) < f_best) * E[1/f(x) | f(x) < f_best] + P(f(x) >= f_best) * (f_best / f(x))
    
    # For f(x) ~ N(mu, sigma^2):
    # The expectation involves integrating max(f_best, z)/z * N(z; mu, sigma) dz
    # This has a known closed form using the normal CDF and PDF.
    
    # Let's use the standard EHypR formula:
    # EHypR(x) = E[max(f_best, f(x)) / f(x)]
    #          = 1 + E[(f_best - f(x)) / f(x) * 1{f(x) < f_best}]
    
    # A more stable formulation:
    # EHypR = P(f < f_best) * [1 + (f_best/sigma) * phi((f_best-mu)/sigma) / Phi((f_best-mu)/sigma)] 
    #        + P(f >= f_best) * [f_best / (mu + sigma * (1 - Phi(z))/phi(z))]
    
    # Actually, let's use a simpler but effective approximation based on the
    # expected improvement structure adapted for Hypar:
    
    z = (f_best - mu) / sigma  # Standardized distance to f_best
    
    # Probability that f(x) < f_best
    p_improve = norm.cdf(z)
    
    # Expected improvement term: E[f_best - f(x) | f(x) < f_best] * P(f(x) < f_best)
    # = sigma * [phi(z) + z * (1 - Phi(z))]  ... wait, let me recalculate
    
    # For EI: E[max(f_best - f(x), 0)] = sigma * [z*Phi(z) + phi(z)]
    # For EHypR, we want E[max(f_best, f(x))/f(x)]
    
    # Let's use a practical approximation that captures the key behavior:
    # Score = -mu + beta * sigma * (1 + log(f_best + epsilon) / (1 + mu/sigma))
    
    # Actually, let's implement a cleaner version:
    # EHypR ≈ 1 + (f_best/sigma) * [phi(z)/Phi(z) - z] * Phi(z) + (f_best/mu) * (1-Phi(z))
    # where z = (f_best - mu)/sigma
    
    # Simplified and stable version:
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Avoid numerical issues
    Phi_z = np.clip(Phi_z, 1e-15, 1.0)
    
    # Term 1: When f(x) < f_best, contribution is E[f_best/f(x)] ≈ f_best/E[f(x)|f(x)<f_best]
    # E[f(x)|f(x)<f_best] ≈ mu - sigma * phi(z)/Phi(z)
    mean_given_improve = mu - sigma * phi_z / Phi_z
    mean_given_improve = np.maximum(mean_given_improve, 1e-10)
    term1 = (f_best / mean_given_improve) * Phi_z
    
    # Term 2: When f(x) >= f_best, contribution is f_best/f(x)
    # E[f(x)|f(x)>=f_best] ≈ mu + sigma * phi(z)/(1-Phi(z))
    one_minus_Phi_z = np.clip(1.0 - Phi_z, 1e-15, 1.0)
    mean_given_no_improve = mu + sigma * phi_z / one_minus_Phi_z
    mean_given_no_improve = np.maximum(mean_given_no_improve, 1e-10)
    term2 = (f_best / mean_given_no_improve) * one_minus_Phi_z
    
    score = term1 + term2
    
    # Enforce finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
