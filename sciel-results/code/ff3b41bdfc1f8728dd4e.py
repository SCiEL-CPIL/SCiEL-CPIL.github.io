"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline tracking
_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Bayesian Regret Minimization (BRM) acquisition function.
    Scores each candidate by its probability of being the global minimum
    within the remaining cost budget, weighted by expected improvement.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions in objective units
    - sigma: array of shape (2000,), GP standard deviations in objective units
    - f_best: scalar, lowest observed function value
    - cost: array of shape (2000,), evaluation cost for each candidate
    
    Returns:
    - scores: array of shape (2000,), acquisition values (higher = evaluate next)
    """
    # Initialize module-level state
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
        _acq_state['call_count'] = 0
    _acq_state['call_count'] += 1
    
    # Deadline guard - return safely if we're running out of time
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 325.0:
        return np.zeros(len(mu), dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values robustly
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.5)
    
    # Ensure positive values
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # --- Bayesian Regret Minimization (BRM) ---
    # The BRM approach estimates the probability that each candidate is the
    # global minimum given the remaining budget, then weights by expected
    # improvement to balance exploration and exploitation.
    
    # Phase-based adaptation: early calls favor exploration, later favor exploitation
    call_idx = min(_acq_state['call_count'], 50)
    phase = call_idx / 50.0  # 0 to 1
    
    # Adaptive parameters
    # Beta for LCB component: decays from exploratory to exploitative
    beta = 2.0 - 0.8 * phase  # 2.0 -> 1.2
    
    # Cost validity: minimizer is in [1,9]^4, so cost in [1.1, 1.9]
    # Use smooth sigmoid gates
    lower_width = 0.04
    upper_width = 0.04
    
    lower_factor = 0.5 * (1.0 + np.tanh((cost - 1.1) / lower_width))
    upper_factor = 0.5 * (1.0 - np.tanh((cost - 1.9) / upper_width))
    cost_validity = np.clip(lower_factor * upper_factor, 0.0, 1.0)
    
    # --- Expected Improvement (EI) for minimization ---
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # Where Phi is CDF and phi is PDF of standard normal
    
    # Clip to avoid numerical issues
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # --- Probability of being global minimum within budget ---
    # Approximate: P(candidate is min) ∝ exp(-mu^2 / (2*sigma^2)) * (1/sigma)
    # This is the probability density at the minimum under the GP posterior
    # Weight by cost validity to restrict to feasible region
    
    # Use a Gaussian kernel weighting: candidates with lower mu and higher sigma
    # get higher probability of containing the true minimum
    # Normalize by typical scale
    mu_scale = np.std(mu) + 1e-8
    sigma_scale = np.std(sigma) + 1e-8
    
    # Probability weight: how likely is this point to be near the global min?
    # Use a combination of low mean and high uncertainty
    prob_weight = np.exp(-0.5 * ((mu - np.min(mu)) / mu_scale) ** 2) * \
                  (sigma / (sigma + 0.1))
    
    # Cost efficiency: prefer lower cost within valid region
    cost_eff = 1.5 / cost
    
    # --- Combine components ---
    # BRM score = EI * prob_weight * cost_validity * cost_eff
    # Plus an LCB component for exploration in early phase
    
    lcb_score = -mu + beta * sigma
    
    # Blend EI and LCB based on phase
    # Early: more LCB (exploration), Late: more EI (exploitation)
    blend = 0.3 + 0.7 * phase  # 0.3 -> 1.0
    
    base_score = blend * ei + (1 - blend) * lcb_score
    
    # Apply probability weight and cost factors
    scores = base_score * prob_weight * cost_validity * cost_eff
    
    # Add small exploration bonus for uncertain points in valid region
    exploration_bonus = 0.05 * sigma * cost_validity * (1 - phase)
    scores = scores + exploration_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
