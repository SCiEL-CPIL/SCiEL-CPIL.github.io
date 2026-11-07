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
    1. GP-UCB score: -mu + beta_t * sigma (encourages exploration of high uncertainty)
    2. Normalized Expected Improvement: encourages exploitation of regions likely to improve f_best
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Handle non-finite f_best
    if not np.isfinite(f_best):
        f_best = 1e6
    
    # Estimate effective time step t from f_best
    # As f_best decreases, t increases, leading to lower beta (more exploitation)
    # Mapping: f_best ~ 100 -> t ~ 1, f_best ~ 10 -> t ~ 4, f_best ~ 1 -> t ~ 17, f_best ~ 0 -> t ~ 21
    f_best_clipped = max(f_best, 0.0)
    t = 1.0 + 20.0 * np.exp(-f_best_clipped / 5.0)
    t = max(t, 1.0)
    
    # Theoretical beta_t = sqrt(2 * t * log(1 + t))
    # This provides a principled schedule for exploration
    beta_t = np.sqrt(2.0 * t * np.log(1.0 + t))
    
    # Clamp beta to reasonable range
    beta_t = np.clip(beta_t, 1.0, 15.0)
    
    # Calculate UCB score for minimization
    # We want to maximize the score.
    # LCB = mu - beta * sigma. We want to minimize LCB.
    # So we maximize -LCB = -mu + beta * sigma.
    ucb_score = -mu + beta_t * sigma
    
    # Calculate Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    z_ei = (f_best - mu) / sigma
    # Clip z to prevent overflow in exp
    z_ei = np.clip(z_ei, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Normalize both scores to [0, 1] for combination
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    if ucb_max - ucb_min > 1e-9:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max - ei_min > 1e-9:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ucb_score)
    
    # Combine normalized scores
    # Weight: More weight on UCB when beta is high (early), more on EI when beta is low (late)
    # beta_t ranges from ~1.0 to ~15.0
    # Normalize beta to [0, 1] for weighting
    beta_norm = (beta_t - 1.0) / 14.0
    beta_norm = np.clip(beta_norm, 0.0, 1.0)
    
    # Weight for UCB: higher when beta is higher (more exploration)
    # Weight for EI: higher when beta is lower (more exploitation)
    w_ucb = 0.4 + 0.4 * beta_norm  # Range [0.4, 0.8]
    w_ei = 1.0 - w_ucb             # Range [0.2, 0.6]
    
    # Ensure weights sum to 1
    total_w = w_ucb + w_ei
    if total_w > 0:
        w_ucb /= total_w
        w_ei /= total_w
    
    # Final score
    score = w_ucb * ucb_norm + w_ei * ei_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
