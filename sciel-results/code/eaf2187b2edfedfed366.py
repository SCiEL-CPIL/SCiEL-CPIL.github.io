import numpy as np
from scipy.stats import norm
import time

_deadline = None
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Probability of Improvement (PI)
    and Expected Improvement (EI) normalized by evaluation cost.
    
    Parameters:
    mu: (2000,) array of mean predictions
    sigma: (2000,) array of standard deviations
    f_best: float, current best observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    (2000,) array of scores (higher is better)
    """
    global _deadline, _start_time
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # 5s safety margin

    if time.time() > _deadline:
        return np.ones(len(mu))

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement for minimization
    # PI = P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * pi + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Hybrid score: Combine PI (robustness) and EI (magnitude)
    # Normalize EI to be comparable to PI? 
    # EI can be large if f_best is far from mu. PI is bounded [0,1].
    # A simple convex combination might work, but scaling is tricky.
    # Let's try: score = PI + alpha * EI
    # Or better: Use EI but dampen it?
    # The current program (PI) scored 1.07. EI scored 1.0.
    # This suggests PI is better for this specific cost-aware setting.
    # Why? Maybe because PI focuses on the *probability* of finding something better,
    # which is a more stable signal when cost varies, whereas EI can be dominated
    # by regions with high uncertainty (high sigma) leading to high EI but potentially
    # low probability of actually being good.
    # However, pure PI ignores how *much* better.
    
    # Let's try a weighted sum where weights depend on the state.
    # If we are close to the optimum (f_best is low, mu is close), PI is good.
    # If we are far, EI might be better to guide us towards lower values.
    
    # Let's try a different structural idea: Cost-Effective Improvement (CEI)
    # CEI = EI / cost
    # But we saw EI/cost scored 1.0.
    
    # What if we use a "Softened" EI?
    # Or what if we combine PI and EI in a way that PI dominates when uncertainty is high?
    
    # Let's look at the "Inspiration 1" which was EI/cost.
    # The current program is PI/cost.
    
    # Let's try a hybrid: score = (PI * 0.5 + EI_normalized * 0.5) / cost
    # How to normalize EI? 
    # Maybe divide EI by the range of f? We don't know the range.
    # Maybe divide by (f_best - min_mu)?
    
    # Alternative: Use PI as the base, but add a small bonus for low mu?
    # score = PI / cost - mu / cost ? No, that's just (PI - mu)/cost.
    
    # Let's try: score = (PI + EI) / cost
    # Since EI >= 0 and PI >= 0, this increases the score for points with high EI.
    # But EI is unbounded. If f_best is -3 and mu is -4, EI is ~1. PI is ~1.
    # If f_best is -3 and mu is -10, EI is ~7. PI is ~1.
    # So (PI + EI) would heavily favor very low mu points.
    # This might be good for exploitation.
    
    # Let's try: score = PI / cost + EI / (cost * scale)
    # Where scale is something like (f_best - np.min(mu)) + eps
    # This normalizes EI relative to the current best possible improvement.
    
    min_mu = np.min(mu)
    range_val = max(f_best - min_mu, 1e-6)
    ei_norm = ei / range_val
    
    # Blend PI and normalized EI
    # Weight 0.7 for PI (since it worked better) and 0.3 for EI_norm
    hybrid = 0.7 * pi + 0.3 * ei_norm
    
    scores = hybrid / (cost + eps)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
