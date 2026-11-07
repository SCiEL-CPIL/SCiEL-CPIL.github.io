"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit

    # Handle edge cases
    if len(mu) == 0:
        return np.array([])
    
    # Ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    n = len(mu)
    
    # GP-LCB (Lower Confidence Bound)
    # We want to minimize f, so we look for low values.
    # The LCB is typically: mu - beta * sigma
    # We want to maximize the score, so we use: -mu + beta * sigma
    # Or equivalently, we can just compute the LCB and return it, 
    # but the task says "higher meaning evaluate next".
    # So we return: -mu + beta * sigma
    
    # Determine beta dynamically.
    # A common heuristic is to start with higher exploration and reduce it as we converge.
    # f_best is the lowest observed value. The known minimum is ~0.398.
    # Let's define a progress metric.
    # If f_best is close to 0.398, we should exploit more (lower beta).
    # If f_best is far, explore more (higher beta).
    
    known_min = 0.397887
    # Normalize the gap to [0, 1] roughly. 
    # Branin values can be up to ~300.
    # Let's say if f_best < 1.0, we are doing well.
    # If f_best > 10, we are doing poorly.
    
    # Use a sigmoid-like mapping or simple linear clamp.
    # Let's use a simple adaptive beta.
    # Base beta around 1.5 to 2.5.
    
    # Calculate a "confidence" level based on how good f_best is.
    # If f_best is very low (close to min), confidence in finding better is low, so explore less?
    # Actually, LCB balances this. 
    # Standard practice: beta can be fixed or time-dependent.
    # Let's use a fixed beta that works well for Branin, or a slightly adaptive one.
    # A beta of ~1.5-2.0 is often good for 2D problems with 2000 candidates.
    
    # Let's try a slightly adaptive beta based on the spread of mu relative to sigma.
    # If sigma is large compared to the range of mu, we should explore more.
    
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    sigma_mean = np.mean(sigma)
    
    # If the uncertainty is high relative to the spread of means, increase beta.
    # Ratio = sigma_mean / (mu_max - mu_min + 1e-9)
    ratio = sigma_mean / (mu_max - mu_min + 1e-9)
    
    # Clamp ratio to avoid extreme values
    ratio = np.clip(ratio, 0.1, 10.0)
    
    # Map ratio to beta
    # If ratio is high, beta should be higher.
    # Let's use beta = 1.0 + ratio
    beta = 1.0 + ratio
    
    # Optional: Adjust beta based on f_best progress
    # If f_best is close to known_min, we might want to be more conservative (lower beta)
    # to exploit the found minimum, unless we suspect we haven't seen the global one.
    # However, LCB naturally handles this if the GP is well-calibrated.
    # Let's stick to the uncertainty-driven beta for now.
    
    # Compute LCB score: -mu + beta * sigma
    # This encourages low mu and high sigma.
    score = -mu + beta * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
