import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive UCB: use higher exploration when uncertainty is high
    # Scale beta based on the typical sigma magnitude
    sigma_mean = np.mean(sigma)
    
    # For Hartmann 3D, the function range is roughly [-3.86, 3.3]
    # Use a beta that balances exploration and exploitation
    # Early on (high sigma_mean), explore more; late on, exploit more
    
    # Try EI with a negative xi for exploration
    # xi controls exploration: negative xi means we're willing to explore
    # even points where mu > f_best
    
    # Let's use a combination: EI with adaptive xi
    # When sigma is high, use more negative xi (more exploration)
    # When sigma is low, use less negative xi (more exploitation)
    
    # Base xi around -0.2, adjusted by sigma magnitude
    # sigma_mean in Hartmann 3D context: typically 0.1-1.0 early, <0.1 late
    xi = -0.3 * np.tanh(sigma_mean / 0.5) - 0.1
    
    target = f_best - xi
    z = (target - mu) / sigma
    scores = (target - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    return scores
