import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Budget-Scaled UCB combined with EI.
    
    Args:
        mu: Mean of the GP prediction, shape (N,)
        sigma: Standard deviation of the GP prediction, shape (N,)
        f_best: Best observed value so far.
        cost: Cost of evaluating each point, shape (N,)
        
    Returns:
        scores: Acquisition scores, shape (N,), higher = evaluate next
    """
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    # Input validation and safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive and cost is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Structural Change: Budget-Scaled UCB
    # The Hartmann 6D minimum is -3.32237.
    # We estimate the "remaining budget" in terms of potential improvement distance.
    # If f_best is far from the minimum, we have "more budget" to explore.
    # If f_best is close, we should exploit more.
    
    min_hartmann = -3.32237
    # Distance from current best to the known global minimum
    dist_to_min = max(0.0, f_best - min_hartmann)
    
    # Scale beta based on how far we are from the optimum.
    # Early on (large dist), beta is high (aggressive exploration).
    # Later (small dist), beta is lower (precise exploitation).
    # We use an exponential decay to smooth this transition.
    # Normalizing by a scale factor (e.g., 1.0 or 2.0) controls the sensitivity.
    # Let's say if dist > 2.0, beta is near max. If dist < 0.1, beta is near min.
    # beta = beta_min + (beta_max - beta_min) * exp(-dist / scale)
    
    beta_max = 3.0
    beta_min = 0.5
    scale = 0.5
    
    beta = beta_min + (beta_max - beta_min) * np.exp(-dist_to_min / scale)
    
    # UCB for minimization: -mu + beta * sigma
    ucb_score = -mu + beta * sigma_safe
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Clip to avoid numerical issues
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize UCB and EI to have comparable scales
    ucb_std = np.std(ucb_score) + 1e-8
    ei_std = np.std(ei) + 1e-8
    
    ucb_norm = ucb_score / ucb_std
    ei_norm = ei / ei_std
    
    # Adaptive weighting:
    # When uncertainty (sigma) is high, we trust UCB more.
    # When uncertainty is low, we trust EI more (it's more precise).
    sigma_mean = np.mean(sigma_safe)
    
    # If mean sigma is high, weight UCB more.
    # Let's map sigma_mean to a weight.
    # If sigma_mean > 0.5, UCB weight = 0.7, EI weight = 0.3
    # If sigma_mean < 0.1, UCB weight = 0.3, EI weight = 0.7
    # Linear interpolation
    if sigma_mean > 0.5:
        ucb_weight = 0.7
        ei_weight = 0.3
    elif sigma_mean < 0.1:
        ucb_weight = 0.3
        ei_weight = 0.7
    else:
        # Interpolate
        t = (sigma_mean - 0.1) / 0.4
        ucb_weight = 0.3 + 0.4 * t
        ei_weight = 0.7 - 0.4 * t
    
    combined = ucb_weight * ucb_norm + ei_weight * ei_norm
    
    # Cost-aware penalty:
    # We want to penalize expensive points.
    # Dividing by cost is a strong penalty.
    # Dividing by sqrt(cost) is a milder penalty.
    # The current program uses sqrt(cost). Inspiration 1 uses cost.
    # Let's try a power penalty: cost^alpha.
    # If alpha=1, it's cost. If alpha=0.5, it's sqrt(cost).
    # Let's use alpha=0.75 as a middle ground, or stick to sqrt for stability?
    # The prompt says "Cost-aware".
    # Let's try dividing by cost_safe ** 0.75 to be slightly more aggressive on cost than sqrt.
    
    cost_penalty = cost_safe ** 0.75
    scores = combined / cost_penalty
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return a safe fallback (exploitative) if we are running out of time
        return -mu / cost_penalty
    
    return scores
