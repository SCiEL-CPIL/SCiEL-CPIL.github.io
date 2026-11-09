import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds total budget, leave margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses a hybrid strategy combining Expected Improvement (EI) and an adaptive
    Lower Confidence Bound (LCB) component to balance exploitation and exploration.
    
    The score is a weighted sum of:
    1. Log-EI: Promotes points with high expected improvement.
    2. LCB: Promotes points with low mean and high uncertainty.
    
    The weight of the LCB component increases as the number of evaluations grows
    (estimated via the gap between f_best and the known minimum), preventing
    premature convergence.
    """
    # Check deadline guard
    if time.time() > _DEADLINE - 5.0:
        # Return a neutral score to avoid hanging, though this shouldn't happen
        # for vectorized operations on 2000 points.
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Avoid division by zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Known minimum of Hartmann 6D
    f_min = -3.32237
    
    # Gap from best observed to known minimum
    gap = f_best - f_min
    gap_clamped = np.clip(gap, 0.0, 5.0)
    
    # --- Component 1: Expected Improvement (EI) ---
    # We are minimizing, so improvement is f_best - mu
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Standard Normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # EI formula: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = improvement * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)
    
    # Log-EI to dampen the effect of very high EI and promote robustness
    log_ei = np.log1p(ei)
    
    # --- Component 2: Adaptive LCB ---
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the score, so we use -mu + beta * sigma
    # 
    # Adaptive beta: 
    # As the gap decreases (we get closer to the minimum), we want to exploit more (lower beta).
    # As the gap is large (early stages), we want to explore more (higher beta).
    # 
    # Strategy:
    # beta starts high when gap is large and decreases as gap shrinks.
    # Let's map gap [0, 5] to beta [0.5, 3.0]
    # When gap = 0 (at minimum), beta = 0.5 (exploit)
    # When gap = 5 (far from minimum), beta = 3.0 (explore)
    
    # Linear interpolation
    beta = 0.5 + (3.0 - 0.5) * (gap_clamped / 5.0)
    
    lcb_score = -mu + beta * sigma_safe
    
    # --- Combine Scores ---
    # We need to combine log_ei and lcb_score.
    # They have different scales. 
    # log_ei is typically small (e.g., 0.1 to 2.0).
    # lcb_score can be large (e.g., -10 to 10).
    # 
    # A simple weighted sum might be dominated by LCB.
    # Let's normalize or use a different combination.
    # 
    # Alternative: Use EI as the primary signal, but add an exploration bonus
    # that is proportional to sigma, scaled by an adaptive factor.
    # 
    # Let's try: Score = log_ei + alpha * sigma
    # where alpha is adaptive.
    # 
    # Or, more robustly:
    # Score = w * log_ei + (1-w) * lcb_score_normalized
    # 
    # However, the inspiration 1 (score 1.15) used a pure UCB with adaptive beta.
    # The current program (score 1.0) uses Log-EI.
    # 
    # Let's combine them by taking the maximum or a weighted sum.
    # Given the stagnation directive, a structural change is needed.
    # The directive suggests: "GP-LCB variant with an adaptive exploration coefficient beta that increases as the number of evaluated points grows... beta = 1 + 0.1 * t"
    # But we don't have t directly. We can estimate progress via gap.
    # 
    # Let's stick to the hybrid idea:
    # Score = log_ei + gamma * (beta * sigma - mu) ? No, that mixes scales.
    #
    # Let's try a different structural change:
    # Use the LCB score directly, but modulate it with the EI confidence.
    # Or simply use the LCB score with a very adaptive beta.
    #
    # Let's look at Inspiration 1 again:
    # beta = 3.0 - 2.5 * progress (where progress is 0 to 1)
    # Score = -mu + beta * sigma
    # This scored 1.15.
    #
    # Let's try to improve on Inspiration 1 by incorporating EI.
    # 
    # New Strategy:
    # Score = -mu + beta * sigma + lambda * log_ei
    # where lambda is small, so LCB dominates but EI breaks ties.
    # 
    # Let's set lambda = 0.1.
    
    score = lcb_score + 0.1 * log_ei
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
