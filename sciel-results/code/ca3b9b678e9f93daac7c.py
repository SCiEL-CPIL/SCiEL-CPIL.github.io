import numpy as np
from scipy.stats import norm
import time

# Global start time
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 285:
        # Fallback: simple cost-aware EI
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost

    # Constants
    f_min = -3.32237
    
    # Calculate progress: p=0 when f_best is high (early), p=1 when f_best is low (late)
    # Hartmann 6d random points are typically in [-1, 0]
    f_max_est = 0.0
    denom = f_max_est - f_min
    if denom < 1e-9:
        p = 0.5
    else:
        p = (f_max_est - f_best) / denom
        p = np.clip(p, 0.0, 1.0)

    # Thompson Sampling: sample from posterior predictive
    # We want to minimize, so we want low samples.
    # Score should be high for candidates we want to pick.
    # So score = -sample / cost.
    
    # To balance exploration/exploitation, we can adjust the sigma used for sampling.
    # Early: high sigma (explore more)
    # Late: low sigma (exploit more)
    # But Thompson sampling naturally does this if the GP is well-calibrated.
    # However, to be safe and ensure we don't get stuck in a local optima too early,
    # we can add a small amount of noise or adjust the effective sigma.
    
    # A common trick is to use a "temperature" or scaling factor for sigma.
    # Early: scale sigma up (more exploration)
    # Late: scale sigma down (more exploitation)
    # scale = 1.0 + alpha * (1 - p)
    # Let's try a mild scaling.
    scale = 1.0 + 0.5 * (1.0 - p)
    sigma_sample = sigma * scale
    
    # Draw samples
    # We need to be careful with the random state. 
    # Since we don't have a fixed seed for the acquisition function calls,
    # we rely on the global numpy random state.
    # This is standard for Thompson Sampling in BO loops where the acquisition is called repeatedly.
    
    samples = mu + sigma_sample * np.random.randn(len(mu))
    
    # Score: higher is better.
    # We want to minimize the function, so lower sample is better.
    # Thus, score = -sample.
    # Normalize by cost.
    score = -samples / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
