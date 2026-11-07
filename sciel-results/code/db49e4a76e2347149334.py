import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin well below the 330s wall-clock kill

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Combines adaptive UCB (for exploration) with Expected Improvement (for exploitation).
    
    The strategy adapts based on progress:
    - Early (f_best far from 3.0): Weight towards UCB to explore the space.
    - Late (f_best close to 3.0): Weight towards EI to exploit the local neighborhood.
    
    The UCB beta is also adaptive, decreasing as we get closer to the optimum.
    """
    # Ensure inputs are arrays of correct type
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Wall-clock guard: never let this call approach the kill limit
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Exploitation-only fallback (finite, fast)
        score = f_best - mu
        return np.clip(score, -1e12, 1e12)

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Known global minimum for Goldstein-Price 2D
    f_min = 3.0
    
    # Calculate the gap to the known global minimum
    gap = max(f_best - f_min, 0.0)
    
    # --- Adaptive UCB Component ---
    # Beta starts high when gap is large, decreases as gap shrinks.
    # Using a smooth decay: beta = base + amp * exp(-gap / scale)
    # When gap=0: beta = base + amp (max exploration if we were somehow at min? No, we exploit)
    # Wait, if gap is small, we want to exploit. So beta should be small.
    # Let's invert the logic or adjust parameters.
    # If gap is large (early), we want high beta.
    # If gap is small (late), we want low beta.
    
    # Let's use: beta = beta_min + (beta_max - beta_min) * exp(-gap / scale)
    # gap=0 -> beta = beta_max? No, that's wrong. We want to explore when we are FAR.
    # So if gap is LARGE, beta should be LARGE.
    # If gap is SMALL, beta should be SMALL.
    
    # Let's try: beta = beta_min + (beta_max - beta_min) * (1 / (1 + gap / scale))
    # gap=0 -> beta = beta_min + (beta_max - beta_min) * 1 = beta_max. (This is wrong, we want min exploration at start? No, we want max exploration at start when we know nothing? But we have 5 points. Usually, early on, uncertainty is high everywhere. UCB with high beta explores high uncertainty. If we are close to opt, uncertainty is low. So maybe beta doesn't need to change as much? Or maybe we want to exploit more when close.)
    
    # Let's look at the current program's logic:
    # beta = 0.5 + 2.5 * np.tanh(gap / 5.0)
    # gap=0 -> beta = 0.5 (Exploit)
    # gap=10 -> beta = 0.5 + 2.5 * 0.99 = ~3.0 (Explore)
    # This makes sense. High gap -> High beta -> Explore.
    # Let's stick to a similar adaptive beta but perhaps smoother or tuned.
    
    # Let's use the tanh approach from the current program as it scored well (1.46)
    beta = 0.5 + 2.5 * np.tanh(gap / 5.0)
    
    # UCB Score for minimization: -mu + beta * sigma
    ucb_score = -mu + beta * sigma_safe
    
    # --- Expected Improvement (EI) Component ---
    # EI is better for exploitation.
    # xi is an exploitation parameter. Small xi encourages improvement.
    # We can adapt xi as well. When gap is small, we want very small xi.
    xi = 0.01 * max(gap, 0.1)
    
    z = (f_best - mu - xi) / sigma_safe
    ei_score = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI scores are finite
    ei_score = np.where(np.isfinite(ei_score), ei_score, 0.0)
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, 0.0)
    
    # --- Blending ---
    # We want to use UCB when gap is large, EI when gap is small.
    # Let's define a weight w_ucb that goes from 1 (pure UCB) to 0 (pure EI) as gap decreases?
    # Or vice versa?
    # Actually, EI is a standard for exploitation. UCB is for exploration.
    # Early on (large gap), we want to explore -> High weight on UCB.
    # Late on (small gap), we want to exploit -> High weight on EI.
    
    # Let's use a sigmoid or similar function to blend.
    # gap is large -> w_ucb close to 1
    # gap is small -> w_ucb close to 0
    
    # Scale gap for blending.
    # If gap < 1, mostly EI. If gap > 5, mostly UCB.
    w_ucb = np.tanh(gap / 2.0)  # 0 to 1
    w_ei = 1.0 - w_ucb
    
    # However, EI scores are typically smaller in magnitude than UCB scores?
    # We might need to normalize or scale them.
    # Let's just try a linear combination first.
    # If they are on different scales, the one with larger magnitude will dominate.
    # EI is bounded by (f_best - min(mu)) roughly.
    # UCB is -mu + beta*sigma.
    
    # To make them comparable, we can normalize them or just use a weighted sum.
    # Let's try: score = w_ucb * ucb_score + w_ei * ei_score
    
    # But wait, if w_ucb=1, score=ucb. If w_ei=1, score=ei.
    # This seems reasonable.
    
    score = w_ucb * ucb_score + w_ei * ei_score
    
    # Ensure scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
