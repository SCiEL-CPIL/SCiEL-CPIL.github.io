import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    z = np.clip(z, -50.0, 50.0)
    
    # Compute Expected Improvement (EI) for minimization
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Compute Quantile-based Expected Improvement (QEI) targeting 95th percentile
    # The 95th percentile of the predictive distribution is mu + 1.645 * sigma
    # For minimization, we want points where the 95th percentile is low
    # This gives a high-confidence lower bound on the function value
    quantile_95 = mu + 1.645 * sigma
    
    # QEI: improvement over f_best using the quantile
    # We expect improvement if quantile_95 < f_best
    # The expected improvement based on the quantile is:
    # QEI = max(0, f_best - quantile_95) + sigma * correction
    # A simpler approach: use the negative of the 95th percentile as a proxy
    # for high-confidence good values, blended with EI
    
    # Alternative QEI formulation: 
    # The probability that f(x) < f_best is Phi(z)
    # The expected value of f(x) given f(x) < f_best is mu - sigma * phi(z)/Phi(z)
    # QEI targets the 95th percentile: we want P(f(x) < f_best) >= 0.95
    # This means z >= 1.645, i.e., (f_best - mu)/sigma >= 1.645
    
    # Compute a score that rewards points where the 95th percentile is below f_best
    # and also considers the expected improvement
    qei_score = np.where(quantile_95 < f_best, f_best - quantile_95, 0.0)
    
    # Normalize both EI and QEI to comparable scales
    ei_min, ei_max = np.min(ei), np.max(ei)
    qei_min, qei_max = np.min(qei_score), np.max(qei_score)
    
    if ei_max - ei_min > 1e-10:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.ones_like(ei)
    
    if qei_max - qei_min > 1e-10:
        qei_norm = (qei_score - qei_min) / (qei_max - qei_min)
    else:
        qei_norm = np.ones_like(qei_score)
    
    # Blend: 70% EI for exploitation, 30% QEI for robust exploitation
    score = 0.7 * ei_norm + 0.3 * qei_norm
    
    # Check time
    if time.time() > deadline - 5.0:
        # Return the best available result quickly
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
