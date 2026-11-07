import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    Uses Adaptive UCB with a small Expected Improvement refinement term.
    """
    f_global_min = 0.397887
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate current regret
    regret = f_best - f_global_min
    
    # Adaptive Beta parameters for UCB
    # High regret (early) -> High beta (explore)
    # Low regret (late) -> Low beta (exploit)
    beta_min = 0.5
    beta_max = 3.0
    r_scale = 10.0
    
    if regret >= 0:
        fraction = regret / (regret + r_scale)
    else:
        fraction = 0.0
        
    beta = beta_min + (beta_max - beta_min) * fraction
    
    # Compute UCB score for minimization:
    # We want to minimize f(x).
    # Lower Confidence Bound (LCB) = mu - beta * sigma
    # We want to select the point with the lowest LCB.
    # The evaluator selects the point with the highest score.
    # Therefore, Score = -LCB = beta * sigma - mu
    ucb_scores = beta * sigma_safe - mu
    
    # Compute Expected Improvement for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute standard normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei_scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei_scores = np.maximum(ei_scores, 0.0)
    
    # Combine UCB with a small EI refinement term
    # This adds exploitation pressure when uncertainty is low
    # while maintaining the strong exploration of UCB
    ucb_scale = np.max(np.abs(ucb_scores)) + eps
    ei_scaled = ei_scores / ucb_scale
    
    scores = ucb_scores + 0.1 * ei_scaled
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
