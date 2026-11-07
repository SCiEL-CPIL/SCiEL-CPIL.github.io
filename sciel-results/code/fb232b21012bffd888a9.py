"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score using Bayesian Regret Minimization (BRM)
    with smooth log-sum-exp based probability of being the new minimum.
    
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
    # Deadline guard
    t0 = time.time()
    deadline = 295.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute improvement: how much better could this point be than f_best?
    improvement = f_best - mu  # positive if mu < f_best
    
    # Compute a confidence-weighted improvement score
    # Use a smooth surrogate for the probability of being the new minimum
    # P(x is new min) ~ softmax over (improvement / sigma) with adaptive temperature
    
    # Adaptive temperature based on the spread of improvements and uncertainties
    imp_std = np.std(improvement) + 1e-12
    sig_mean = np.mean(sigma_safe) + 1e-12
    
    # Temperature: scale by the typical uncertainty relative to improvement spread
    # This makes the softmax adaptive to the current state of the GP
    temp = imp_std / (sig_mean + 1e-12)
    temp = np.clip(temp, 0.1, 5.0)
    
    # Compute z-scores for the probability of being the minimum
    # Larger improvement relative to uncertainty -> higher probability
    z = improvement / (sigma_safe * temp + 1e-12)
    
    # Softmax via log-sum-exp for numerical stability
    z_max = np.max(z)
    log_sum_exp = z_max + np.log(np.sum(np.exp(z - z_max)) + 1e-30)
    log_prob_min = z - log_sum_exp
    prob_min = np.exp(log_prob_min)
    
    # Cost-aware BRM: weight by probability of being minimum, divided by cost
    brm_score = prob_min / cost_safe
    
    # Add exploration component: reward high-uncertainty regions
    # This helps escape local optima and explore unexplored areas
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-15:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Scale exploration relative to BRM score magnitude
    brm_max = np.max(brm_score)
    if brm_max > 1e-15:
        exploration_weight = 0.15 * brm_max
    else:
        exploration_weight = 0.0
    
    exploration = exploration_weight * sigma_norm
    
    # Combine BRM and exploration
    scores = brm_score + exploration
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() - t0 > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
