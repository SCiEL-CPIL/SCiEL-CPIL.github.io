import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.maximum(sigma, 1e-10)
    
    # Estimate number of evaluations t from f_best
    # Initial 5 points are uniform random on [-4.5, 4.5]^2
    # Beale function at random points: values can range widely
    # As we find better points, f_best decreases
    # We'll use a heuristic to estimate t based on how good f_best is
    
    # For Beale function on [-4.5, 4.5]^2:
    # The minimum is 0 at (3, 0.5)
    # Typical random values are quite high (tens to hundreds)
    # After a few good evaluations, f_best drops significantly
    
    # Estimate t: we'll use the fact that with GP, better f_best means more evaluations done
    # Simple heuristic: if f_best < 1, assume t >= 10
    # If f_best < 0.1, assume t >= 15
    # Otherwise, estimate based on magnitude
    
    if f_best < 0.01:
        t_est = 20
    elif f_best < 0.1:
        t_est = 15
    elif f_best < 1.0:
        t_est = 10
    elif f_best < 10.0:
        t_est = 7
    else:
        t_est = 5
    
    # GP-UCB with time-varying beta_t = sqrt(2 * t * log(1 + t))
    t = float(t_est)
    beta_t = np.sqrt(2.0 * t * np.log(1.0 + t))
    
    # UCB score: -mu + beta_t * sigma (since we're minimizing)
    ucb_score = -mu + beta_t * sigma
    
    # Normalized Expected Improvement
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # For minimization, improvement is when f < f_best
    
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # Normalize EI to have similar scale as UCB term
    # Scale EI by a factor to balance with UCB
    # The UCB term has scale related to sigma, EI also related to sigma
    # Use a small weight for EI to maintain exploitation pressure without dominating
    
    ei_weight = 0.5
    combined_score = ucb_score + ei_weight * ei
    
    # Ensure finite values
    combined_score = np.nan_to_num(combined_score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return combined_score
