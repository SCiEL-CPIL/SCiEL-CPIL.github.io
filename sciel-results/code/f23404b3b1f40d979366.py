"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline tracking
_acq_start_time = None


def acquisition(mu, sigma, f_best, cost):
    """
    Dynamic UCB acquisition with cost-awareness and deadline guard.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    global _acq_start_time
    
    # Deadline guard: if we've been running too long, return safe fallback
    if _acq_start_time is not None:
        elapsed = time.time() - _acq_start_time
        # Reserve safety margin; if elapsed > 280s, return quick fallback
        if elapsed > 280:
            # Fallback: pure exploitation (pick lowest mu)
            return -mu.copy()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Guard against zero or negative costs
    cost_safe = np.maximum(cost, 1e-9)
    
    # Dynamic exploration weight based on remaining budget
    # cost is the cost of each candidate; total budget is 30
    # f_best is the best observed value
    # We estimate remaining budget from the mean cost of candidates
    mean_cost = np.mean(cost_safe)
    # Estimate how many more evaluations we can afford
    # The total cost budget is 30, and we've spent some already
    # We don't know exactly how much we've spent, but we can use the cost vector
    # as a proxy: lower cost candidates are more affordable
    
    # Strategy: UCB with dynamic beta that decreases as we get more certain
    # Use the ratio of sigma to mu scale to adapt
    mu_scale = np.std(mu) if np.std(mu) > 1e-9 else 1.0
    sigma_scale = np.std(sigma) if np.std(sigma) > 1e-9 else 1.0
    
    # Normalize sigma relative to the uncertainty scale
    sigma_norm = sigma / (sigma_scale + 1e-9)
    
    # Dynamic beta: start high (exploration) and decrease
    # Use the spread of mu as a proxy for how much we've learned
    # When mu spread is small relative to initial, we're more certain
    mu_range = np.ptp(mu) if np.ptp(mu) > 1e-9 else 1.0
    
    # Beta that adapts: higher when uncertainty is high relative to improvement
    # Base beta around 1.0, modulated by cost efficiency
    # Cheaper candidates get slightly more exploration weight
    cost_mean = np.mean(cost_safe)
    cost_ratio = cost_safe / (cost_mean + 1e-9)
    
    # UCB: mu - beta * sigma / cost (for minimization, we want low mu and high sigma)
    # Actually for minimization: we want to minimize mu + beta * sigma (penalty for uncertainty)
    # But UCB for minimization is: mu - beta * sigma (we prefer uncertain low values)
    # Wait: for minimization, UCB is mu - beta * sigma (exploit low mu, explore high sigma)
    # The acquisition should be high for good candidates
    # Good for minimization: low mu, high sigma (uncertainty)
    # So score = -mu + beta * sigma / cost (normalize by cost)
    
    # Adaptive beta: scale by remaining budget proxy
    # If mean cost is low, we can afford more exploration
    # If mean cost is high, we should exploit more
    base_beta = 1.0
    # Modulate by cost: cheaper candidates get more exploration
    beta = base_beta / cost_safe
    
    # UCB score for minimization
    score = -mu + beta * sigma
    
    # Additional: penalize very high cost candidates more aggressively
    # Use a soft penalty that scales with cost
    cost_penalty = 0.1 * (cost_safe - 1.0)  # mild penalty for expensive candidates
    score = score - cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
