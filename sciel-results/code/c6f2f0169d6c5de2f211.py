"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for tracking call count and elapsed time
_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score combining BRM-style probability
    of improvement with cost-adjusted LCB dynamics.
    
    The BRM component uses a softmax over confidence-weighted improvement
    to estimate the probability each candidate is the new minimum.
    The LCB component uses a budget-aware beta that decreases as the
    cost limit of 30 is approached, forcing exploitation when budget
    is scarce.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best observed value so far (lowest, since we minimize)
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x scaled to [0,1])
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    # Initialize module-level state
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    _acq_state['call_count'] += 1
    
    # Deadline guard
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 305.0:
        return np.zeros(len(mu), dtype=np.float64)
    
    t0 = time.time()
    deadline = 295.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    n_calls = _acq_state['call_count']
    
    # === BRM Component: probability of being the new minimum ===
    improvement = f_best - mu  # positive if mu < f_best
    
    # Adaptive temperature based on spread of improvements and uncertainty
    imp_std = np.std(improvement) + 1e-12
    mean_sigma = np.mean(sigma_safe) + 1e-12
    temperature = 0.5 * (imp_std + mean_sigma) + 1e-12
    
    # Confidence-weighted improvement z-scores
    z_improve = improvement / (temperature + sigma_safe * 0.1)
    
    # Softmax for probability of being minimum (log-sum-exp for stability)
    z_max = np.max(z_improve)
    exp_z = np.exp(z_improve - z_max)
    prob_min = exp_z / (np.sum(exp_z) + 1e-12)
    
    # Cost-aware BRM score
    brm_score = prob_min / cost_safe
    
    # === Cost-Adjusted LCB Component ===
    # Budget-aware beta: as total cost approaches 30, reduce exploration
    # Estimate remaining budget fraction from call count heuristic
    # Total budget is 30, initial 5 points cost ~5*1.5=7.5, so ~22.5 remaining
    # Each call adds ~1.5 cost on average, so ~15 more calls possible
    # Use call count to estimate budget depletion
    if n_calls <= 3:
        beta_base = 2.5
    elif n_calls <= 6:
        beta_base = 1.8
    elif n_calls <= 10:
        beta_base = 1.2
    elif n_calls <= 14:
        beta_base = 0.7
    else:
        beta_base = 0.3
    
    # Additional budget-aware scaling: if we estimate we're near the limit,
    # further reduce beta. Use a smooth transition.
    # Estimated total cost so far: initial 5 points + n_calls evaluations
    # Each eval costs ~1 + mean(x in [0,1]) ≈ 1.5 on average
    est_total_cost = 7.5 + n_calls * 1.5
    budget_fraction = max(0.0, min(1.0, (30.0 - est_total_cost) / 22.5))
    
    # Scale beta by budget fraction with a floor to maintain some exploration
    beta = beta_base * (0.3 + 0.7 * budget_fraction)
    
    # LCB for minimization: mu - beta * sigma (lower is better)
    # We want to maximize the score, so use negative LCB
    lcb = mu - beta * sigma_safe
    lcb_score = -lcb  # higher = better (lower predicted value)
    
    # Cost-adjusted LCB
    lcb_cost = lcb_score / cost_safe
    
    # Normalize both components
    brm_max = np.max(brm_score) if np.max(brm_score) > 1e-15 else 1.0
    lcb_max = np.max(lcb_cost) if np.max(lcb_cost) > 1e-15 else 1.0
    
    brm_norm = brm_score / brm_max
    lcb_norm = lcb_cost / lcb_max
    
    # Blend BRM and LCB with time-varying weights
    # Early: more LCB exploration, Late: more BRM exploitation
    if n_calls <= 5:
        w_brm = 0.4
        w_lcb = 0.6
    elif n_calls <= 10:
        w_brm = 0.55
        w_lcb = 0.45
    elif n_calls <= 15:
        w_brm = 0.7
        w_lcb = 0.3
    else:
        w_brm = 0.8
        w_lcb = 0.2
    
    scores = w_brm * brm_norm + w_lcb * lcb_norm
    
    # Add exploration bonus for high-uncertainty regions in early stages
    if n_calls <= 8:
        sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 1e-12 else 1.0
        sigma_norm = sigma_safe / sigma_max
        # Weight by how far from f_best (less weight if already promising)
        mean_gap = np.maximum(0, f_best - mu)
        mean_gap_max = np.max(mean_gap) if np.max(mean_gap) > 1e-12 else 1.0
        gap_weight = 1.0 - 0.2 * (mean_gap / mean_gap_max)
        exploration_bonus = 0.15 * sigma_norm * gap_weight / cost_safe
        scores += exploration_bonus
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    if time.time() - t0 > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
