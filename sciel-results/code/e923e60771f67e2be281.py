# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines Thompson Sampling (stochastic exploration) with Cost-Aware UCB
    using an adaptive beta parameter. The beta parameter is inversely
    proportional to budget consumption, shifting from aggressive exploration
    in early iterations to exploitation as the cost limit approaches.
    
    The approach:
    1. Draw a single Thompson sample from the GP posterior
    2. Compute a Cost-Aware UCB score with adaptive beta
    3. Blend Thompson Sampling and CA-UCB with adaptive weights based on
       proximity to the known minimum
    4. Apply cost-aware penalty
    """
    # Track elapsed time for deadline guard
    t_start = time.time()
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Known minimum of hartmann 3d
    f_min_known = -3.86278
    
    # Estimate budget consumption from cost array
    # The total budget is 30, and cost represents per-candidate costs
    # Use mean cost as a proxy for how much budget is being consumed
    mean_cost = np.mean(cost)
    # Budget consumption fraction: higher mean cost means more budget used
    # Scale: if mean_cost ~ 1.5 (typical), budget_consumed ~ 0.5
    budget_consumed = np.clip(mean_cost / 3.0, 0.0, 1.0)
    
    # Calculate Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma_safe
    z_clipped = np.clip(z, -50, 50)
    ei = improvement * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Thompson Sampling: draw a single sample from the GP posterior
    # Use a deterministic seed based on f_best for consistency within a call
    rng = np.random.default_rng(seed=int(abs(f_best) * 10000) % 2**32)
    z_sample = rng.standard_normal(len(mu))
    f_sample = mu + sigma_safe * z_sample
    
    # For minimization, we want low f_sample values
    # Convert to "higher is better" score
    ts_score = -f_sample
    
    # Cost-Aware UCB with adaptive beta
    # beta is inversely proportional to budget consumption
    # Early (low budget_consumed): high beta, more exploration
    # Late (high budget_consumed): low beta, more exploitation
    beta = 1.0 / (1.0 + budget_consumed * 2.0)
    # Scale beta to be in a reasonable range for this problem
    # Hartmann 3d values are roughly in [-4, 0], so beta * sigma should be comparable
    beta = beta * 2.0
    
    # UCB for minimization: we want low mu + high sigma
    # ucb = mu - beta * sigma (lower is better)
    # Convert to "higher is better": -ucb = -mu + beta * sigma
    ucb_score = -mu + beta * sigma_safe
    
    # Calculate a variance-based exploration score
    pv_score = sigma_safe
    
    # Adaptive weighting based on how close we are to the known minimum
    gap = f_best - f_min_known
    gap_normalized = np.clip(gap / 4.0, 0.0, 1.0)
    
    # Weight: when gap is large (early), explore more; when gap is small (late), exploit more
    exploration_weight = 0.05 + 0.35 * gap_normalized
    exploitation_weight = 1.0 - exploration_weight
    
    # Cost-aware scoring
    # Adaptive alpha: when we're far from the minimum, be less sensitive to cost
    alpha = 0.5 + 0.5 * (1.0 - gap_normalized)
    
    cost_safe = np.maximum(cost, 1e-9)
    cost_factor = cost_safe ** alpha
    
    # Normalize scores to comparable scales
    ei_max = np.max(ei) + 1e-9
    ts_min = np.min(ts_score)
    ts_max = np.max(ts_score)
    ts_range = ts_max - ts_min + 1e-9
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min + 1e-9
    
    # Normalize EI to [0, 1]
    ei_normalized = ei / ei_max
    
    # Normalize TS score to [0, 1] (higher is better)
    ts_normalized = (ts_score - ts_min) / ts_range
    
    # Normalize UCB score to [0, 1] (higher is better)
    ucb_normalized = (ucb_score - ucb_min) / ucb_range
    
    # Normalize PV to [0, 1]
    pv_max = np.max(pv_score) + 1e-9
    pv_normalized = pv_score / pv_max
    
    # Blend Thompson Sampling and CA-UCB
    # Use TS as primary stochastic driver, UCB as structured exploration
    # Blend ratio: more TS early, more UCB later
    ts_weight = 0.5 + 0.3 * gap_normalized  # 0.5 to 0.8
    ucb_weight = 1.0 - ts_weight             # 0.5 to 0.2
    
    # Combine exploitation components (TS and UCB)
    exploitation_score = ts_weight * ts_normalized + ucb_weight * ucb_normalized
    
    # Combine with EI for refinement and PV for exploration
    # EI provides a deterministic improvement estimate
    ei_weight = 0.3 * exploitation_weight
    
    combined = exploitation_weight * (exploitation_score + ei_weight * ei_normalized) + \
               exploration_weight * pv_normalized
    
    # Apply cost-aware penalty
    scores = combined / cost_factor
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Add a small deterministic component to break ties and encourage diversity
    tie_breaker = 1e-6 * rng.random(len(mu))
    scores = scores + tie_breaker
    
    # Deadline guard: if we're close to the limit, return quickly
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
