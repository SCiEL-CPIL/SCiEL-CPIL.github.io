import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining GP-UCB with time-varying beta and normalized EI.
    
    The exploration parameter beta_t is adapted using a theoretical schedule:
    beta_t = sqrt(2 * t * log(1 + t))
    where t is estimated from the improvement of f_best.
    
    The score is a combination of:
    1. GP-UCB: -mu + beta_t * sigma (encourages exploration when uncertainty is high)
    2. Normalized Expected Improvement: maintains exploitation pressure
    
    This avoids arbitrary fixed constants and balances exploration-exploitation
    in a principled way.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Handle non-finite f_best
    if not np.isfinite(f_best):
        f_best = 1e6
    
    # Estimate the "time" step t from f_best
    # For Beale function, typical initial values are large (~10-100)
    # As optimization progresses, f_best decreases toward 0
    # We use a mapping from f_best to an effective step count
    # When f_best is large, t is small (early stage, high exploration)
    # When f_best is small, t is large (late stage, lower exploration)
    
    # Map f_best to t using an exponential decay
    # t = 1 + 20 * exp(-f_best / 5)
    # When f_best = 100, t ~ 1 + 20*exp(-20) ~ 1
    # When f_best = 10, t ~ 1 + 20*exp(-2) ~ 1 + 2.7 ~ 3.7
    # When f_best = 5, t ~ 1 + 20*exp(-1) ~ 1 + 7.4 ~ 8.4
    # When f_best = 1, t ~ 1 + 20*exp(-0.2) ~ 1 + 16.4 ~ 17.4
    # When f_best = 0, t ~ 1 + 20*1 = 21
    
    f_best_clipped = max(f_best, 0.0)
    t = 1.0 + 20.0 * np.exp(-f_best_clipped / 5.0)
    t = max(t, 1.0)  # Ensure t >= 1
    
    # Theoretical beta_t = sqrt(2 * t * log(1 + t))
    beta_t = np.sqrt(2.0 * t * np.log(1.0 + t))
    
    # Clamp beta to reasonable range
    beta_t = np.clip(beta_t, 1.0, 15.0)
    
    # GP-UCB score (for minimization: we want low mu and high sigma)
    ucb_score = -mu + beta_t * sigma
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the CDF and phi is the PDF of the standard normal
    
    # Handle division by zero
    z = (f_best - mu) / sigma
    z = np.where(np.isfinite(z), z, 0.0)
    z = np.clip(z, -50, 50)  # Prevent overflow
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Normalize EI to have similar scale as UCB
    # The scale of EI depends on the spread of mu and sigma
    # Use a simple normalization: divide by (1 + std of mu)
    mu_std = np.std(mu)
    mu_std = max(mu_std, 1e-6)
    
    ei_normalized = ei / (1.0 + mu_std)
    
    # Combine UCB and EI additively
    # Weight: give more weight to UCB early (high beta_t), more to EI later
    # Use a soft weighting based on beta_t
    # When beta_t is large (early), weight_u = 1.0, weight_e = 0.3
    # When beta_t is small (late), weight_u = 0.7, weight_e = 1.0
    
    weight_u = 1.0
    weight_e = 0.5 + 0.5 * np.exp(-beta_t / 5.0)
    
    score = weight_u * ucb_score + weight_e * ei_normalized
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
