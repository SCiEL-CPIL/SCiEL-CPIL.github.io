import numpy as np
import time
from scipy.stats import norm

# Global state to track progress for adaptive weighting
_state = {
    'start_time': None,
    'f_best_history': [],
    'step_count': 0
}

def acquisition(mu, sigma, f_best):
    """
    Deterministic time-adaptive hybrid acquisition function.
    
    Strategy:
    1. Compute Expected Improvement (EI) for exploitation.
    2. Compute GP-UCB score for exploration.
    3. Dynamically blend them using a weight `w` that decays over time 
       and in response to improvements in f_best.
    
    This addresses the Beale function's structure by:
    - Early on (high w): UCB dominates, ensuring we explore the entire 
      [-4.5, 4.5]^2 domain to find the narrow valley.
    - Later (low w): EI dominates, allowing precise localization of the 
      minimum within the valley.
    
    The dynamic adjustment of `w` based on f_best improvement rate helps 
    avoid premature convergence if the initial random points are poor, 
    and accelerates convergence if a good region is found early.
    """
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
        _state['step_count'] = 0
        _state['f_best_history'] = []
    
    # Deadline guard: if we are close to the 330s limit, force a very 
    # exploitative mode to ensure we return a valid, finite result quickly
    elapsed = time.time() - _state['start_time']
    if elapsed > 325:
        # Return pure exploitation (lowest mu) to be safe and fast
        return -mu

    _state['step_count'] += 1
    _state['f_best_history'].append(f_best)
    
    # Compute EI
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # Note: We are minimizing, so improvement is f_best - mu.
    # If mu > f_best, EI is 0.
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    # Avoid overflow in exp for large negative z
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    # Ensure EI is non-negative (numerical noise can make it slightly negative)
    ei = np.maximum(ei, 0)
    
    # Compute UCB
    # UCB = -mu + beta * sigma
    # We need to choose beta. A common choice is sqrt(2 * log(t)) or similar.
    # Here we use a dynamic beta that starts high and decays, but is modulated by 
    # how much f_best has improved.
    t = _state['step_count']
    # Base beta: starts around 3-4, decays slowly
    beta = 2.0 * np.sqrt(np.log(t + 1) + 1.0)
    
    # Dynamic adjustment: If f_best has improved significantly recently, 
    # we can afford to be more exploitative (lower beta).
    # If stagnation is detected, we increase beta to explore more.
    if len(_state['f_best_history']) >= 2:
        recent_improvement = _state['f_best_history'][-2] - f_best
        # If recent_improvement > 0, we improved.
        # Normalize improvement by the scale of the problem (roughly)
        # Beale function range is roughly 0 to 15+.
        # A threshold of 0.1 is a reasonable "significant" improvement.
        if recent_improvement > 0.1:
            beta *= 0.8  # Exploit more
        elif recent_improvement < 0.01:
            beta *= 1.2  # Explore more (stagnation)
    
    ucb = -mu + beta * sigma_safe
    
    # Dynamic blending weight w
    # w = weight for UCB (exploration), 1-w = weight for EI (exploitation)
    # w should start high (e.g., 0.8) and decay to low (e.g., 0.2)
    # Decay based on step count
    w = 0.8 * np.exp(-t / 10.0) + 0.2
    
    # Also modulate w by the "quality" of the current best.
    # If f_best is still very high (e.g., > 5), we need more exploration.
    if f_best > 5.0:
        w = min(1.0, w + 0.3)
    elif f_best < 0.5:
        w = max(0.0, w - 0.2)
        
    # Normalize scores to prevent scale issues between EI and UCB
    # EI is in objective units. UCB is in objective units + beta*sigma.
    # They are comparable, but let's ensure we don't have extreme outliers.
    
    # Hybrid score
    scores = w * ucb + (1 - w) * ei
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
