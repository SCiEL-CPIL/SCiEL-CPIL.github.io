import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Structure-aware, cost-adjusted acquisition function for Styblinski-Tang 2D.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of acquisition scores (higher is better).
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are running out of time, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 280:  # Safety margin before 300s limit
        # Return a simple, fast fallback: just prioritize low cost and low mu
        mu_safe = np.asarray(mu, dtype=float)
        cost_safe = np.asarray(cost, dtype=float)
        mu_safe = np.nan_to_num(mu_safe, nan=0.0, posinf=1e6, neginf=-1e6)
        cost_safe = np.nan_to_num(cost_safe, nan=1.0, posinf=1e6, neginf=1e-6)
        cost_safe = np.maximum(cost_safe, 1e-6)
        return -mu_safe / cost_safe

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Ensure finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6)
    
    # Avoid division by zero in cost
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Structure-aware bias:
    # The Styblinski-Tang 2D function has its global minimum at x* = (-2.90345, -2.90345) in [-5, 5]^2.
    # However, the problem states that the minimizer is moved to a uniform point in the inner 80% of each range.
    # "inner 80% of each range" means the range is reduced by 10% on each side.
    # Original range: [-5, 5]. Width = 10.
    # Inner 80%: 10 * 0.8 = 8.
    # New range for the minimum: [-5 + 1, 5 - 1] = [-4, 4].
    # So the true minimum s_k is uniformly distributed in [-4, 4]^2.
    
    # Since we don't know s_k, we cannot directly target it.
    # However, the problem statement says "styblinski tang 2d (min -78.3323)" and "minimizer is moved to a uniform point in the inner 80%".
    # This implies the function is f(x - s_k).
    # The original minimum is at x_orig = -2.90345 * ones(2).
    # The new minimum is at x_new = x_orig + s_k.
    # Wait, the problem says "f_k(x) = f(x - s_k)".
    # If f has minimum at x_orig, then f(x - s_k) has minimum at x - s_k = x_orig => x = x_orig + s_k.
    # s_k is uniform in [-4, 4]^2? Or is the *location* of the minimum uniform in [-4, 4]^2?
    # "whose minimizer is moved to a uniform point in the inner 80% of each range"
    # This means the new minimizer m_k is uniform in [-4, 4]^2.
    # So m_k = x_orig + s_k => s_k = m_k - x_orig.
    
    # We don't know m_k. But we can use the GP to estimate it.
    # However, the directive suggests a "deterministic, structure-aware score that prioritizes the known global minimum region".
    # Since the minimum is moved, the "known" global minimum is no longer fixed.
    # BUT, the distribution of the minimum is uniform in [-4, 4]^2.
    # The mean of this distribution is 0.
    # So, a reasonable deterministic prior is to prioritize the center of the domain (0,0) in the original coordinates,
    # which corresponds to the center of the possible minimum locations.
    
    # Let's convert the candidates to the original coordinate space?
    # No, the GP is trained on scaled coordinates [0,1]^2.
    # The cost is "1 + mean(x scaled to [0, 1])".
    # The candidates are in scaled space [0,1]^2.
    # The original coordinates x_orig = -5 + 10 * x_scaled.
    # The center of the domain in scaled space is (0.5, 0.5), which corresponds to x_orig = (0, 0).
    # Since the minimum is uniform in [-4, 4]^2 (original), its mean is (0,0) (original).
    # So, we should bias towards (0.5, 0.5) in scaled space.
    
    # Let's compute the distance from each candidate to the center (0.5, 0.5) in scaled space.
    # We don't have the coordinates, only mu, sigma, cost.
    # Wait, "No coordinates are passed."
    # This is a critical constraint. We cannot compute the distance to the center because we don't have the x coordinates.
    
    # Re-reading the prompt: "mu, sigma and cost have shape (2000,). Return 2000 finite scores... No coordinates are passed."
    # This means we cannot use the position of the candidates to bias towards the center.
    # We can only use mu, sigma, and cost.
    
    # So, the "structure-aware" bias must be derived from mu and sigma.
    # The GP mean mu(x) approximates f(x).
    # The minimum of the function is around -78.33.
    # So, we can bias towards candidates where mu is low.
    # This is already done in the LCB/UCB term.
    
    # What other structure can we use?
    # The cost is "1 + mean(x scaled to [0, 1])".
    # This means candidates with higher scaled coordinates are more expensive.
    # The minimum is uniform in [-4, 4]^2 (original), which is [0.1, 0.9] in scaled space.
    # The center (0.5, 0.5) has cost 1 + 0.5 = 1.5.
    # The corners (0,0) and (1,1) have cost 1 and 2, respectively.
    # The average cost in the inner 80% box [0.1, 0.9]^2 is 1 + 0.5 = 1.5.
    
    # Since we cannot access coordinates, we cannot directly bias towards the center.
    # However, we can use the GP's prediction.
    # If the GP has learned the function, mu will be low near the minimum.
    # So, a strong bias towards low mu is appropriate.
    
    # Let's refine the acquisition function:
    # 1. Use a strong exploitation term: -mu / cost.
    # 2. Use an exploration term: beta * sigma / cost.
    # 3. Add a "prior" term that encourages exploration of regions where the GP is uncertain but the cost is low?
    #    Or, since the minimum is likely in the inner 80%, and we don't know where, we should explore broadly but cheaply.
    
    # Let's try a different approach:
    # The problem is that the current LCB with beta=2.0 might be too exploratory or too exploitative.
    # Let's use a dynamic beta that decreases as f_best approaches the minimum.
    
    # f_best is the lowest observed value.
    # The minimum is -78.3323.
    # improvement = f_best - (-78.3323)
    # If improvement is large, we are far from the optimum, so explore more.
    # If improvement is small, we are close, so exploit more.
    
    min_val = -78.3323
    improvement = max(f_best - min_val, 1e-6)
    
    # Beta should be larger when improvement is large.
    # Let's use beta = np.clip(2.0 * np.log(improvement), 1.0, 5.0)
    # But improvement can be huge initially.
    # Let's scale it.
    # Typical f_best starts around 0 or positive.
    # improvement ~ 80.
    # log(80) ~ 4.4.
    # beta ~ 8.8. Clipped to 5.0.
    # Later, improvement ~ 10.
    # log(10) ~ 2.3.
    # beta ~ 4.6.
    # Later, improvement ~ 1.
    # log(1) = 0.
    # beta ~ 1.0.
    
    beta = np.clip(2.0 * np.log(improvement), 1.0, 5.0)
    
    # LCB = mu - beta * sigma
    # Score = -LCB / cost = (-mu + beta * sigma) / cost
    
    lcb = mu - beta * sigma
    score = -lcb / cost
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return score
