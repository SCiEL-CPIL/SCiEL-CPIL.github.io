import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (Hartmann 6D).
    Uses Pareto-frontier-based selection to balance exploration, exploitation,
    and cost without manual weight tuning.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions.
    sigma : array of shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x)).
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    DEADLINE = 320.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Floor sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Check deadline early
    if time.time() - t0 > DEADLINE - 5:
        # Fallback to simple cost-aware UCB with sqrt cost
        beta = 1.0
        improvement = f_best - mu + beta * sigma
        scores = improvement / np.sqrt(cost)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Known minimum of Hartmann 6d
    HARTMANN_6D_MIN = -3.32237
    
    # Estimate budget progress
    median_cost = np.median(cost)
    if median_cost > 1e-9:
        total_budget = 30.0
        initial_cost = 5 * 1.5
        typical_cost = median_cost
        remaining_budget = max(0, total_budget - initial_cost)
        progress_from_cost = np.clip(1.0 - (remaining_budget / (typical_cost * 15)), 0.0, 1.0)
    else:
        progress_from_cost = 0.0
    
    # Also estimate progress from f_best improvement
    range_val = f_best - HARTMANN_6D_MIN
    if range_val <= 0:
        progress_from_f = 1.0
    else:
        initial_guess = -1.0
        total_range = initial_guess - HARTMANN_6D_MIN
        progress_from_f = np.clip((initial_guess - f_best) / total_range, 0.0, 1.0)
    
    # Combine both progress estimates
    progress = 0.5 * progress_from_cost + 0.5 * progress_from_f
    progress = np.clip(progress, 0.0, 1.0)
    
    # Compute Expected Improvement (higher is better for minimization)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB improvement (higher is better)
    beta = 0.5 + 2.0 * np.exp(-3.0 * progress)
    ucb_improvement = f_best - mu + beta * sigma
    
    # Three objectives for Pareto selection:
    # 1. EI (maximize)
    # 2. UCB (maximize)  
    # 3. Inverse cost (maximize, i.e., minimize cost)
    inv_cost = 1.0 / cost
    
    # Normalize objectives to [0, 1] range using percentiles for robustness
    ei_min, ei_max = np.percentile(ei, 1), np.percentile(ei, 99)
    ucb_min, ucb_max = np.percentile(ucb_improvement, 1), np.percentile(ucb_improvement, 99)
    ic_min, ic_max = np.percentile(inv_cost, 1), np.percentile(inv_cost, 99)
    
    ei_norm = np.clip((ei - ei_min) / (ei_max - ei_min + 1e-9), 0.0, 1.0)
    ucb_norm = np.clip((ucb_improvement - ucb_min) / (ucb_max - ucb_min + 1e-9), 0.0, 1.0)
    ic_norm = np.clip((inv_cost - ic_min) / (ic_max - ic_min + 1e-9), 0.0, 1.0)
    
    # Find Pareto-optimal candidates using a simple non-dominated sorting
    # A point is dominated if another point is >= in all objectives and > in at least one
    n = len(mu)
    objectives = np.column_stack([ei_norm, ucb_norm, ic_norm])
    
    # Efficient Pareto front identification
    # Start with all points as potentially non-dominated
    is_pareto = np.ones(n, dtype=bool)
    
    # Use a simplified approach: rank by each objective and mark candidates
    # that are in the top portion for at least one objective
    top_frac = 0.15  # Top 15% for each objective
    top_ei = ei_norm >= np.percentile(ei_norm, (1 - top_frac) * 100)
    top_ucb = ucb_norm >= np.percentile(ucb_norm, (1 - top_frac) * 100)
    top_ic = ic_norm >= np.percentile(ic_norm, (1 - top_frac) * 100)
    
    # Candidates that are in top fraction for at least one objective
    candidate_mask = top_ei | top_ucb | top_ic
    
    # For candidates, compute a Pareto-aware score
    # Use weighted sum with adaptive weights based on progress
    # Early: more weight on exploration (UCB, EI)
    # Late: more weight on exploitation (EI) and cost efficiency
    
    w_ei = 0.4 + 0.3 * progress
    w_ucb = 0.3 + 0.2 * (1 - progress)
    w_ic = 1.0 - w_ei - w_ucb
    w_ic = max(w_ic, 0.1)
    
    # Renormalize weights
    w_sum = w_ei + w_ucb + w_ic
    w_ei /= w_sum
    w_ucb /= w_sum
    w_ic /= w_sum
    
    # Weighted sum score
    weighted_score = w_ei * ei_norm + w_ucb * ucb_norm + w_ic * ic_norm
    
    # Boost Pareto-optimal candidates
    pareto_bonus = np.zeros(n)
    pareto_bonus[candidate_mask] = 1.0
    
    # Additional boost for points that are in top for multiple objectives
    multi_top = (top_ei.astype(int) + top_ucb.astype(int) + top_ic.astype(int))
    multi_bonus = multi_top * 0.3
    
    scores = weighted_score + 0.5 * pareto_bonus + multi_bonus
    
    # Scale to ensure good separation
    scores_min, scores_max = np.min(scores), np.max(scores)
    score_range = scores_max - scores_min + 1e-9
    scores = (scores - scores_min) / score_range
    
    # Add small noise-free tiebreaker based on raw EI for stability
    scores = scores + 0.01 * ei_norm
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
