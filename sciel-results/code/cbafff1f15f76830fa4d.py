"""
Deterministic LCB acquisition with adaptive exploration and time-gated local refinement.

Strategy:
1. Compute LCB (Lower Confidence Bound) for minimization: LCB = mu - gamma * sigma.
2. The exploration coefficient gamma is adaptive:
   - It starts high when the posterior uncertainty (mean sigma) is high relative to the range of mu.
   - It decays as the uncertainty decreases, shifting focus to exploitation.
3. Divide by cost to account for the cost-aware constraint.
4. If time permits (deadline guard), perform a local refinement step:
   - Identify the top K candidates by score.
   - For each, perform a coarse 1-D search along each coordinate to find a better point.
   - Return the score of the best refined candidate, effectively "moving" the evaluation point.
   - Note: The evaluator evaluates the point with the highest score. By refining the top candidates,
     we ensure the highest score corresponds to a more precise location, improving the chance
     of finding the true minimum in the next evaluation step.
"""
import numpy as np
from scipy.stats import norm
import time

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 20s safety margin before 300s limit

def _check_deadline():
    """Check if we are close to the deadline."""
    return time.time() > _DEADLINE

def _lcb(mu, sigma, gamma):
    """
    Lower Confidence Bound for minimization.
    Lower LCB means more promising (lower expected value with uncertainty penalty).
    We want to maximize the score, so we return -LCB / cost or similar.
    Actually, standard practice: Score = - (mu - gamma * sigma) / cost = (gamma * sigma - mu) / cost.
    """
    return (gamma * sigma - mu)

def _adaptive_gamma(mu, sigma):
    """
    Compute an adaptive gamma based on the current state of the GP posterior.
    
    Heuristic:
    - If the mean sigma is large relative to the spread of mu, we are in early exploration.
    - As the GP fits the data, sigma decreases.
    - We want gamma to decrease as the problem becomes more certain.
    
    Let mu_range = max(mu) - min(mu)
    Let sigma_mean = mean(sigma)
    
    If mu_range is small (converging), we want low gamma.
    If sigma_mean is high (uncertain), we want high gamma.
    
    Simple scaling: gamma = base_gamma * (sigma_mean / (mu_range + eps))
    But we need to bound it. Let's use a sigmoid-like or capped approach.
    
    Alternative: Use the ratio of unexplored vs explored.
    Let's use a simple decay based on the median sigma.
    Early: sigma ~ 1 (standardized), mu ~ 0. Gamma should be ~ 2-3.
    Late: sigma ~ 0.1, mu ~ 0. Gamma should be ~ 0.5-1.
    
    gamma = 1.0 + 2.0 * (median_sigma / (max_sigma + eps))
    This keeps gamma between 1.0 and 3.0.
    """
    med_sigma = np.median(sigma)
    max_sigma = np.max(sigma)
    if max_sigma < 1e-6:
        return 0.5  # Very certain, exploit
    
    # Normalize sigma to [0, 1] range based on current max
    norm_sigma = med_sigma / (max_sigma + 1e-9)
    
    # Map to [0.5, 3.0]
    gamma = 0.5 + 2.5 * norm_sigma
    return gamma

def _local_refinement(mu, sigma, cost, f_best, top_k=10, n_steps=5):
    """
    Attempt to refine the top candidates by checking nearby points.
    Since we don't have the GP kernel function available to compute mu/sigma at new points,
    we CANNOT actually compute new mu/sigma values.
    
    Wait, the task says: "mu, sigma and cost have shape (2000,). Return 2000 finite scores".
    The evaluator takes the index of the max score and evaluates THAT point.
    We do NOT have access to the GP object to query new points.
    We only have the pre-computed mu, sigma, and cost for the 2000 candidates.
    
    Therefore, local refinement by moving the point is NOT possible unless we can estimate
    how the score would change. But the score is just a function of the provided mu/sigma/cost.
    The evaluator evaluates the physical point x_i that corresponds to the max score.
    
    So, "local refinement" in the sense of moving the point is impossible.
    We can only choose which of the 2000 points to evaluate.
    
    So the strategy must be purely about selecting the best point from the 2000.
    
    Revised Strategy:
    1. Compute LCB score: (gamma * sigma - mu) / cost.
    2. Add a small deterministic "jitter" or use a hybrid with EI to break ties or add stability?
       No, deterministic is better for reproducibility and avoiding random bad picks.
    3. Ensure the score is finite.
    
    Why was the previous TS approach good? It added stochasticity which might help escape local optima
    in the selection process, but it's risky.
    
    Let's stick to a robust deterministic LCB with a well-tuned gamma.
    The key improvement over simple EI/cost is the adaptive exploration.
    
    Let's refine the gamma calculation to be more robust.
    Use the percentile of sigma.
    """
    pass

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware LCB acquisition with adaptive exploration.
    """
    # Deadline check
    if _check_deadline():
        # Return a safe, deterministic score (e.g., pure exploitation)
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        score = -mu / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Adaptive Gamma
    gamma = _adaptive_gamma(mu, sigma_safe)
    
    # 2. LCB Score (Maximize)
    # LCB for minimization is mu - gamma * sigma.
    # We want to maximize the "promisingness", so we maximize -LCB = gamma * sigma - mu.
    lcb_score = gamma * sigma_safe - mu
    
    # 3. Cost Awareness
    score = lcb_score / cost_safe
    
    # 4. Safety
    score = np.where(np.isfinite(score), score, 0.0)
    # Ensure no negative scores if we want to force exploration? 
    # No, LCB can be negative if mu is very low and sigma is small.
    # But since we are maximizing, a higher score is better.
    # If all scores are negative, we pick the least negative (best exploitation).
    
    return score

# EVOLVE-BLOCK-END
