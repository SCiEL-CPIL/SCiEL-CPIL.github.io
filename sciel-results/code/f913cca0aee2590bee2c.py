"""Cost-aware acquisition for three-hump-camel 2D BO using Adaptive Weighted Normalized EI and Efficiency."""
import time
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive weighting of normalized EI and efficiency.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value (lower is better)
    cost : ndarray, shape (2000,)
        Evaluation cost for each candidate point
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Calculate Efficiency (Inverse Cost)
    cost_safe = np.maximum(cost, 1e-10)
    efficiency = 1.0 / cost_safe
    
    # Normalize EI to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-10:
        ei_norm = np.ones_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
        
    # Normalize Efficiency to [0, 1]
    eff_min = np.min(efficiency)
    eff_max = np.max(efficiency)
    eff_range = eff_max - eff_min
    if eff_range < 1e-10:
        eff_norm = np.ones_like(efficiency)
    else:
        eff_norm = (efficiency - eff_min) / eff_range
        
    # Determine weight w based on convergence state (f_best)
    # If f_best is very low (close to 0), we are in exploitation phase -> prioritize EI (w close to 1)
    # If f_best is high, we are in exploration phase -> prioritize Efficiency/Cost (w close to 0.5)
    # Three-hump camel min is 0.
    # Let's map f_best to a weight.
    # f_best ~ 0 -> w = 0.9
    # f_best ~ 1.0 -> w = 0.5
    # f_best > 1.0 -> w = 0.5
    
    # Simple sigmoid mapping
    # x = f_best
    # w = 0.5 + 0.4 * sigmoid(-k * x)
    # If x=0, sigmoid(0)=0.5, w=0.7. 
    # If x=1, sigmoid(-k). If k=5, sigmoid(-5)=0.0067, w=0.5.
    # If x=0, we want higher w.
    # Let's use w = 0.5 + 0.4 * np.exp(-2.0 * f_best)
    # f_best=0 -> w=0.9
    # f_best=1 -> w=0.5+0.4*0.135=0.55
    # f_best=2 -> w=0.5+0.4*0.018=0.507
    # This seems reasonable.
    
    w = 0.5 + 0.4 * np.exp(-2.0 * f_best)
    
    # Combine
    score = w * ei_norm + (1.0 - w) * eff_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard
    if time.time() > deadline:
        pass
        
    return score
