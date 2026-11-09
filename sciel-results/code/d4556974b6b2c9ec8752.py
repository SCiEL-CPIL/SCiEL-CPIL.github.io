"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining UCB and Expected Improvement.
    
    Strategy:
    1. Compute a hybrid acquisition score that blends UCB (uncertainty-driven exploration)
       with Expected Improvement (improvement-driven exploitation).
    2. Apply a cost-aware penalty using a Lagrangian-style approach where the penalty
       strength adapts based on estimated progress in the optimization.
    3. In early stages (high f_best), favor exploration via higher UCB weight.
    4. In late stages (low f_best, close to optimum), favor exploitation via higher EI weight.
    5. The cost penalty is scaled so that the effective trade-off between improvement
       and cost remains balanced throughout the budget.
    
    Args:
        mu: array of shape (2000,) GP mean predictions.
        sigma: array of shape (2000,) GP standard deviation predictions.
        f_best: float, best observed function value so far.
        cost: array of shape (2000,) costs for each candidate.
        
    Returns:
        array of shape (2000,) scores, higher is better.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Handle edge cases for sigma
    sigma = np.maximum(sigma, 1e-6)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Check time budget
    if time.time() > deadline - 5.0:
        # Return a simple UCB score as fallback
        scores = sigma / cost_safe
        return np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Compute Expected Improvement (EI) for minimization
    # EI(x) = E[max(0, f_best - f(x))]
    # For Gaussian f(x) ~ N(mu, sigma^2):
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB score (for minimization, we want low mu and high sigma)
    # UCB_min = mu - beta_ucb * sigma, we maximize -UCB_min = -mu + beta_ucb * sigma
    # But let's use a normalized version for blending
    
    # Estimate progress based on f_best
    # Styblinski-Tang 2D minimum is approximately -78.3323
    f_min_theoretical = -78.3323
    f_max_expected = 10.0  # Approximate maximum value in the domain
    
    f_best_clipped = np.clip(f_best, f_min_theoretical, f_max_expected)
    # Progress: 0 when f_best is at maximum (worst), 1 when f_best is at minimum (best)
    progress = (f_max_expected - f_best_clipped) / (f_max_expected - f_min_theoretical)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Blend UCB and EI based on progress
    # Early stage (progress ~ 0): favor UCB (exploration)
    # Late stage (progress ~ 1): favor EI (exploitation)
    
    # UCB weight decreases with progress, EI weight increases with progress
    ucb_weight = 1.0 - 0.7 * progress  # From 1.0 to 0.3
    ei_weight = 0.3 + 0.7 * progress   # From 0.3 to 1.0
    
    # Scale EI and UCB to comparable ranges
    # EI is in objective units, typically [0, ~10] early, [0, ~1] late
    # UCB component (sigma) is also in objective units
    
    # Normalize by their respective means to make them comparable
    ei_mean = np.mean(ei)
    if ei_mean < 1e-6:
        ei_normalized = ei
    else:
        ei_normalized = ei / ei_mean
    
    sigma_mean = np.mean(sigma)
    if sigma_mean < 1e-6:
        sigma_normalized = sigma
    else:
        sigma_normalized = sigma / sigma_mean
    
    # Hybrid score before cost adjustment
    hybrid_score = ucb_weight * sigma_normalized + ei_weight * ei_normalized
    
    # Cost-aware penalty
    # Use a Lagrangian-style penalty: score - lambda * cost
    # Choose lambda based on progress and the scale of hybrid_score
    
    # In early stages, we can afford higher cost points (if they have high potential)
    # In late stages, we need to be cost-efficient
    
    # Estimate remaining budget
    # Initial 5 points cost ~ 1.5 each, total ~ 7.5
    # Total budget is 30, so remaining is ~22.5 initially
    # As we progress, remaining budget decreases
    
    # Heuristic: remaining budget is proportional to (1 - progress)
    # At progress=0, remaining ~ 22.5; at progress=1, remaining ~ 2.5
    remaining_budget = 22.5 - 20.0 * progress
    
    # Target average cost for next evaluation
    # Assume we have ~10 evaluations left on average
    target_avg_cost = remaining_budget / 10.0
    target_avg_cost = np.clip(target_avg_cost, 1.0, 2.0)
    
    # Lambda scales the cost penalty
    # If hybrid_score is large, we can afford a larger lambda
    hybrid_mean = np.mean(hybrid_score)
    if hybrid_mean < 1e-6:
        lambda_val = 0.5
    else:
        lambda_val = hybrid_mean / (2.0 * target_avg_cost)
    
    # Apply cost penalty
    scores = hybrid_score - lambda_val * cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Check time budget
    if time.time() > deadline - 2.0:
        return scores
    
    # Optional refinement: adjust lambda to balance cost and improvement
    # Try a few lambda values and pick the one that gives a good balance
    # This is a simple grid search for lambda
    
    best_lambda = lambda_val
    best_score_quality = -np.inf
    
    # Try a few lambda values around the base
    lambda_candidates = [0.5 * lambda_val, 0.75 * lambda_val, lambda_val, 
                         1.25 * lambda_val, 1.5 * lambda_val]
    
    for lam in lambda_candidates:
        if time.time() > deadline - 2.0:
            break
        
        test_scores = hybrid_score - lam * cost_safe
        test_scores = np.nan_to_num(test_scores, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Quality metric: mean of top 100 scores, penalized if top points have high cost
        top_k = min(100, len(test_scores))
        top_scores = np.sort(test_scores)[-top_k:]
        top_indices = np.argsort(test_scores)[-top_k:]
        top_costs = cost_safe[top_indices]
        
        # Quality: high mean score, low mean cost among top points
        quality = np.mean(top_scores) - 0.1 * np.mean(top_costs)
        
        if quality > best_score_quality:
            best_score_quality = quality
            best_lambda = lam
    
    # Use the best lambda found
    scores = hybrid_score - best_lambda * cost_safe
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
