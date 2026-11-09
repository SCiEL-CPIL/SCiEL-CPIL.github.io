import numpy as np
import time
from scipy.stats import norm

# Global state for tracking the optimization step within a seed
_state = {
    "call_count": 0,
    "prev_f_best": None,
    "start_time": None,
    "total_bo_steps": 25  # 30 total evaluations - 5 initial = 25 BO steps
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Expected Improvement (EI) with an adaptive exploration bonus that
    transitions from exploratory to exploitative behavior.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (N,)
    sigma (np.ndarray): Std dev of the GP posterior, shape (N,)
    f_best (float): Best observed value so far (minimum).
    
    Returns:
    np.ndarray: Scores for each candidate, higher is better.
    """
    global _state
    
    # Initialize start time if not set
    if _state["start_time"] is None:
        _state["start_time"] = time.time()
    
    # Check deadline guard: if we are close to the limit, return a safe fallback
    elapsed = time.time() - _state["start_time"]
    if elapsed > 325.0:
        mu = np.asarray(mu, dtype=np.float64)
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Detect if this is a new seed (new optimization run)
    is_new_seed = False
    if _state["prev_f_best"] is not None:
        if f_best > _state["prev_f_best"] + 1e-9 or _state["call_count"] >= _state["total_bo_steps"]:
            is_new_seed = True
            
    if is_new_seed:
        _state["call_count"] = 0
    else:
        _state["call_count"] += 1
        
    _state["prev_f_best"] = f_best
    
    # Ensure step count is within valid range [0, 24]
    current_step = min(_state["call_count"], _state["total_bo_steps"] - 1)
    
    # Calculate progress in the optimization loop [0, 1]
    progress = current_step / max(_state["total_bo_steps"] - 1, 1)
    
    # Ensure inputs are clean
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle any non-finite values in sigma or mu
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Structural change: Expected Improvement (EI) acquisition
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # This is the expected improvement over f_best under the GP posterior
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z for numerical stability
    z = np.clip(z, -10, 10)
    
    # EI computation
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is zero where there's no expected improvement, so add a small
    # exploration bonus in early iterations to encourage diversity
    # The bonus fades as we gain confidence
    exploration_weight = 0.1 * (1.0 - progress) ** 2
    
    # Exploration bonus: reward points with high posterior uncertainty
    # that are not already well-explored (i.e., where mu is not already close to f_best)
    # Use a smooth function that peaks at moderate uncertainty
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        sigma_rel = sigma_safe / sigma_max
        # Bell-shaped exploration bonus centered at moderate uncertainty
        exploration_bonus = exploration_weight * sigma_rel * (1.0 - sigma_rel)
    else:
        exploration_bonus = np.zeros_like(sigma_safe)
    
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
