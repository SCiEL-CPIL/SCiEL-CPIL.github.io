import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For Gaussian posterior: EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    
    # z-score: how many std devs below f_best each candidate's mean is
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # Expected Improvement for minimization
    # When z > 0, mu < f_best, so improvement is likely
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Also compute Lower Confidence Bound (LCB) as an alternative
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to minimize, so lower LCB is better
    # Score for LCB: -LCB (higher score = lower LCB)
    beta = 2.0  # Exploration-exploitation tradeoff
    lcb = mu - beta * sigma
    lcb_score = -lcb
    
    # Hybrid: combine EI and LCB scores
    # Normalize both to have zero mean and unit variance for fair combination
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std > 1e-10:
        ei_norm = (ei - ei_mean) / ei_std
    else:
        ei_norm = np.zeros_like(ei)
    
    lcb_mean = np.mean(lcb_score)
    lcb_std = np.std(lcb_score)
    if lcb_std > 1e-10:
        lcb_norm = (lcb_score - lcb_mean) / lcb_std
    else:
        lcb_norm = np.zeros_like(lcb_score)
    
    # Weighted combination: 0.7 * LCB + 0.3 * EI
    # LCB emphasizes exploration, EI emphasizes exploitation
    scores = 0.7 * lcb_norm + 0.3 * ei_norm
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
