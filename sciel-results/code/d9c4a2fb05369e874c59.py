import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Computes a hybrid acquisition score combining time-varying GP-UCB and normalized EI.
    
    Parameters:
    mu: array of shape (N,) posterior means
    sigma: array of shape (N,) posterior standard deviations
    f_best: float, the best observed value so far
    
    Returns:
    scores: array of shape (N,) scores (higher is better)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Safety checks for finite values
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        return np.zeros_like(mu)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Infer the number of evaluations t from the posterior.
    # In a GP with standardized targets and small noise, the posterior variance 
    # at a point is roughly proportional to the number of points seen.
    # A more robust heuristic: 
    # If we assume the GP prior variance is 1 (standardized), and noise is small,
    # the posterior variance at a point far from training data is ~1.
    # Near training data, it's small.
    # However, we don't have access to training data.
    # We can estimate t by looking at the spread of sigma.
    # Alternatively, a common heuristic in these loops is that t increases linearly.
    # Since we don't know t directly, we can use a proxy.
    # The prompt says "at each step... until 30 evaluations".
    # We can estimate t based on the "confidence" of the model.
    # A simple heuristic: if sigma is very small everywhere, t is large.
    # Let's use a dynamic beta that starts small and grows.
    # Since we don't have t, we can use a heuristic based on the range of sigma.
    # Or, we can just use a fixed schedule if we assume the loop is deterministic.
    # But the function is called 25 times (30 - 5 initial).
    # We can't know which call it is.
    
    # Let's try a different approach: Use the median of sigma to gauge "exploration".
    # If median sigma is high, we are early (t is small). If low, we are late (t is large).
    # This allows the acquisition function to adapt without knowing t explicitly.
    
    med_sigma = np.median(sigma_safe)
    
    # Heuristic mapping of median sigma to t (number of evaluations)
    # If med_sigma is close to 1 (prior variance), t is small.
    # If med_sigma is close to 0, t is large.
    # Let's map med_sigma in [0, 1] to t in [1, 30]
    # t_est = 1 + (1 - med_sigma) * 29
    # This is a rough estimate.
    
    # Actually, a simpler and more robust method for "time-varying" without t:
    # Use a beta that depends on the uncertainty itself.
    # But the directive asks for beta_t = sqrt(2t log(1+t)).
    # Since we can't get t, let's use a proxy for t.
    # The number of evaluations is bounded by 30.
    # We can estimate t by assuming that the reduction in variance is related to t.
    # Let's assume t is proportional to (1 / median_sigma).
    # If median_sigma = 1, t ~ 1. If median_sigma = 0.1, t ~ 10.
    # t_est = 1 / max(med_sigma, 0.1)
    # Cap t_est at 30.
    
    t_est = 1.0 / max(med_sigma, 0.05)
    t_est = min(t_est, 30.0)
    t_est = max(t_est, 1.0)
    
    # Calculate beta_t
    # beta_t = sqrt(2 * t * log(1 + t))
    beta_t = np.sqrt(2.0 * t_est * np.log(1.0 + t_est))
    
    # Calculate UCB score (for minimization, we want low mu and high sigma)
    # UCB = mu - beta_t * sigma  (Lower is better)
    # But we want higher score to be better.
    # So Score_UCB = -mu + beta_t * sigma
    
    score_ucb = -mu + beta_t * sigma_safe
    
    # Calculate EI score
    z = (f_best - mu) / sigma_safe
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    ei = (f_best - mu) * Phi + sigma_safe * phi
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI and UCB to combine them
    # We want to avoid arbitrary constants.
    # Let's normalize both to [0, 1] based on their current distribution.
    
    def normalize_scores(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-9:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    score_ucb_norm = normalize_scores(score_ucb)
    score_ei_norm = normalize_scores(ei)
    
    # Combine: 50% UCB, 50% EI
    # This is a fixed blend, but the components are adaptive.
    # To make it more structural, we can weight them based on t_est.
    # Early on (small t), explore more (UCB). Late on (large t), exploit more (EI).
    # weight_ucb = 1 / (1 + t_est/10)
    # weight_ei = 1 - weight_ucb
    
    weight_ucb = 1.0 / (1.0 + t_est / 10.0)
    weight_ei = 1.0 - weight_ucb
    
    scores = weight_ucb * score_ucb_norm + weight_ei * score_ei_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
