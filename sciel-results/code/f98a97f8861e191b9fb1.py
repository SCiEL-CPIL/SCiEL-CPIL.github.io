import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 2.0 # 2 seconds safety margin for this specific call
    
    # Constants
    f_min = -3.32237
    
    # Ensure sigma is finite and positive
    sigma_safe = np.where(sigma > 0, sigma, 1e-6)
    
    # Log-Expected Improvement (Log-EI) for minimization
    # We want to maximize the score.
    # Improvement is defined as f_best - f(x).
    # Since we are minimizing, lower f(x) is better.
    # Let Z = (f_best - mu) / sigma.
    # E[log(1 + max(0, f_best - f))] is approximated by log(1 + E[max(0, f_best - f)])
    # E[max(0, f_best - f)] = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution.
    
    # Calculate Z
    Z = (f_best - mu) / sigma_safe
    
    # Calculate Phi(Z) and phi(Z)
    # Using standard normal CDF and PDF
    # scipy.stats.norm.cdf and pdf are available but we can use numpy if we want to avoid imports? 
    # The prompt says SciPy is available. But to be safe and fast, let's use scipy if imported or fallback.
    # The current program does not import scipy. I will add the import.
    
    try:
        from scipy.stats import norm
        Phi_Z = norm.cdf(Z)
        phi_Z = norm.pdf(Z)
    except ImportError:
        # Fallback to numpy approximations if scipy is not available (though it should be)
        # Approximation for Phi
        Phi_Z = 0.5 * (1 + np.tanh(Z / np.sqrt(2))) # Rough approx
        phi_Z = np.exp(-0.5 * Z**2) / np.sqrt(2 * np.pi)
        
    # Expected Improvement (EI)
    EI = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Clip EI to be positive (should be >= 0 theoretically, but numerical errors might occur)
    EI = np.maximum(EI, 0.0)
    
    # Log-EI: log(1 + EI)
    # This encourages finding points with high expected improvement, 
    # but the log scale dampens the effect of very high EI, promoting more exploration 
    # than pure EI, and is more robust than UCB's linear combination.
    
    scores = np.log1p(EI)
    
    # Check for NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline check (though this is instant)
    if time.time() > deadline:
         # Return best effort if somehow slow (unlikely)
         pass
         
    return scores
