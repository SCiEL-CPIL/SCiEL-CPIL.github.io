import numpy as np

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw a sample from the GP posterior
    # Note: f_best is not used in standard TS, but we keep it in signature.
    # Some variants use f_best to bias the sample, but standard TS is unbiased.
    
    samples = np.random.normal(mu, sigma_safe)
    
    # We want to minimize the function, so higher score is better for the evaluator
    # if the score corresponds to a lower function value.
    scores = -samples
    
    # Safety: replace non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
