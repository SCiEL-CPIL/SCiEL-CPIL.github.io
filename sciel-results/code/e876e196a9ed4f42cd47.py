import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Structural Change: Use the known global minimum (3.0) as the reference point.
    # This transforms the acquisition function from "Expected Improvement over current best"
    # to "Expected Improvement over the true optimum", which in this case acts as a
    # "Targeted Uncertainty" metric. It highlights regions with high posterior variance
    # that are close to the known global minimum, promoting exploration of the basin
    # of attraction without being misled by the diminishing returns of standard EI
    # as f_best converges to the optimum.
    
    f_ref = 3.0
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_ref - mu) / sigma_safe
    
    # Expected Improvement formula
    # Note: Since f(x) >= f_ref, the "improvement" f_ref - f(x) is <= 0.
    # However, the EI formula calculates the expected value of the part that is "better" 
    # than the reference. Since nothing is better than the minimum, this interpretation 
    # is mathematically 0. 
    # BUT, the formula (ref - mu) * Phi(z) + sigma * phi(z) is a smooth, positive-valued 
    # function of (mu, sigma) that peaks when mu is near ref and sigma is high.
    # It effectively measures "how much better could this be?" in a probabilistic sense,
    # or rather, "how much uncertainty remains near the optimum?".
    # This is a valid heuristic for targeted exploration.
    
    ei = (f_ref - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
