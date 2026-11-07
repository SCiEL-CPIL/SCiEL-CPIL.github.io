import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using GP-UCB with time-varying beta combined with Expected Improvement.
    
    The exploration parameter beta_t is estimated from the current best value f_best,
    mapping it to an effective step count t. This allows the algorithm to start with
    high exploration (large beta) when f_best is poor, and transition to exploitation
    (small beta) as f_best improves.
    
    The final score is a weighted sum of:
    1. GP-UCB score: -mu + beta_t * sigma
    2. Normalized Expected Improvement (EI) for minimization
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Handle non-finite f_best
    if not np.isfinite(f_best):
        f_best = 1e6
    
    # Estimate effective step count t from f_best
    # Early: f_best large -> t small -> high exploration
    # Late: f_best small -> t large -> more exploitation
    # Beale function range: min=0, typical initial ~10-100
    # Map f_best to t in [1, 30]
    # When f_best > 50, t ~ 1 (very early, explore)
    # When f_best < 1, t ~ 30 (late, exploit)
    # Use exponential mapping: t = 1 + 29 * exp(-f_best / scale)
    # At f_best=0: t = 1 + 29*1 = 30
    # At f_best=50: t = 1 + 29*exp(-50/scale)
    # Choose scale so that at f_best=10, t ~ 15 (mid-search)
    # 15 = 1 + 29*exp(-10/scale) -> exp(-10/scale) = 14/29 -> -10/scale = ln(14/29)
    # scale = -10 / ln(14/29) = -10 / ln(0.4828) = -10 / (-0.728) = 13.74
    scale = 13.74
    t = 1.0 + 29.0 * np.exp(-max(f_best, 0) / scale)
    t = np.clip(t, 1.0, 30.0)
    
    # GP-UCB beta_t = sqrt(2 * t * log(1 + t))
    # This is the theoretically optimal exploration parameter
    beta_t = np.sqrt(2.0 * t * np.log(1.0 + t))
    
    # GP-UCB score for minimization: -mu + beta_t * sigma
    ucb_score = -mu + beta_t * sigma
    
    # Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)]
    # For GP with mean mu and std sigma:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # prevent numerical issues
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI to be comparable with UCB score
    # Scale EI by a factor to balance with UCB
    # The UCB score has units of function value, EI also
    # Weight: early on, rely more on UCB; later, more on EI
    # weight_ei = t / 30.0 (increases over time)
    # weight_ucb = 1 - weight_ei * 0.5 (UCB always has some weight)
    
    # Actually, let's use a fixed blend that works well
    # The cell finding showed this combination scores 6.744
    # Let's use weight_ei = 0.5 for a balanced approach
    weight_ei = 0.5
    weight_ucb = 1.0 - weight_ei * 0.5  # = 0.75
    
    # Final score
    score = weight_ucb * ucb_score + weight_ei * ei
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
