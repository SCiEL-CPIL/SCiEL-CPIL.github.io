import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks for division
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score for minimization: z = (f_best - mu) / sigma
    # If mu < f_best, z is positive (likely improvement)
    # If mu > f_best, z is negative (likely worse)
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    # Phi(z) is the Probability of Improvement (PoI)
    # phi(z) is the PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is always non-negative.
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PoI)
    # PoI = Phi(z)
    # This is a dimensionless confidence measure in [0, 1].
    poi = cdf_z
    poi = np.clip(poi, 0.0, 1.0)
    
    # Structural Idea: Adaptive Blend
    # The "EI * (1 + PoI)" formulation from the cell findings suggests
    # boosting the Expected Improvement by a factor that increases with
    # the Probability of Improvement.
    #
    # Why this helps:
    # 1. EI already accounts for the magnitude of improvement.
    # 2. PoI accounts for the confidence.
    # 3. Multiplying EI by (1 + PoI) effectively doubles the weight of
    #    high-confidence improvements (PoI ~ 1) compared to low-confidence
    #    ones (PoI ~ 0.5 or lower).
    # 4. This creates a sharper preference for regions where the GP is both
    #    confident and predicts improvement, which is often better than pure
    #    EI which might be attracted to high-variance regions with moderate
    #    expected improvement.
    # 5. It is a structural change from a linear blend (alpha*EI + (1-alpha)*PoI)
    #    to a multiplicative modulation, changing the shape of the acquisition
    #    landscape.
    
    # Compute the modulated EI
    score = ei * (1.0 + poi)
    
    # Cost-aware normalization
    # Dividing by cost prioritizes cheaper evaluations.
    # Cost is in [1, 2].
    score = score / cost_safe
    
    # Ensure finite outputs
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 290:
        return np.zeros_like(mu)
        
    return score
