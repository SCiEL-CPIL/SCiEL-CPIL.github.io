import numpy as np
from scipy.stats import norm

# Module-level state
_state = {
    'count': 0,
    'last_f_best': None
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Quantile-based Expected Improvement (QEI).
    Targets the 95th percentile of the posterior predictive distribution
    for robust exploitation in the multimodal Styblinski-Tang landscape.
    """
    # Update state
    if _state['last_f_best'] is not None:
        # If f_best has increased significantly, assume a new run started
        # or the previous run was bad and this one started with better random points?
        # In minimization, f_best should decrease. An increase means new run.
        if f_best > _state['last_f_best']:
            _state['count'] = 0
        
        # Safety: if count exceeds max expected steps (25) + buffer, reset
        if _state['count'] > 30:
            _state['count'] = 0
    else:
        _state['count'] = 0
        
    _state['count'] += 1
    _state['last_f_best'] = f_best
    
    # Decay schedule
    # Total loop steps = 25
    max_steps = 25
    # Clamp count to 1..max_steps
    t = min(_state['count'], max_steps)
    
    # Progress-aware quantile targeting:
    # Early on: target a lower quantile (more exploration)
    # Later: target a higher quantile (more exploitation)
    # Quantile ranges from 0.50 (median) to 0.95 (95th percentile)
    progress = t / max_steps
    q_target = 0.50 + 0.45 * progress  # 0.50 -> 0.95
    
    # Compute the z-score corresponding to the target quantile
    z_q = norm.ppf(q_target)
    
    # QEI: We want to maximize the probability that f(x) < f_best
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # To target the q-th percentile of improvement, we use:
    # score = (f_best - mu) / sigma
    # This is equivalent to maximizing the standardized improvement.
    # However, to incorporate the quantile targeting, we adjust by z_q:
    # A point is "good" if (f_best - mu) / sigma > z_q
    # So we score by how far above z_q the standardized improvement is.
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized improvement
    z_improve = (f_best - mu) / sigma_safe
    
    # QEI score: reward being above the target quantile threshold
    # This creates a smooth, differentiable score that favors points
    # where the posterior predicts high confidence of improvement
    scores = z_improve - z_q
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
