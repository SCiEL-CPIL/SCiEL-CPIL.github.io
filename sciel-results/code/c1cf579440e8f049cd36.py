"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 250.0  # 300s limit, leave 50s margin
    
    # Numerical stability: clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standard EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 1e-12)
    
    # LogEI to mitigate dominance of outliers
    log_ei = np.log(1.0 + ei)
    
    # UCB-like term for exploration: -mu + kappa * sigma
    # For minimization, lower mu is better, so we want to reward low mu.
    # UCB score = -mu + kappa * sigma
    # Let's try a combination of LogEI and a scaled UCB term
    # The UCB term helps explore areas with high uncertainty even if EI is low
    
    # Check time
    if time.time() > deadline:
        # Return LogEI as fallback
        scores = np.where(np.isfinite(log_ei), log_ei, 0.0)
        return scores
    
    # Try different kappa values for UCB and combine with LogEI
    # A common approach is to use a mixture or weighted sum
    
    # Let's try a simple weighted combination:
    # score = alpha * LogEI + beta * UCB_normalized
    
    # Normalize UCB to have similar scale as LogEI
    # UCB = -mu + kappa * sigma
    # Let's try kappa = 0.5, 1.0, 2.0
    
    # Actually, let's try a different approach: use the "Expected Improvement"
    # but with a slight modification to encourage more exploration early on.
    
    # One idea: use a temperature-like parameter that decays
    # But we don't know the iteration count.
    
    # Let's try a simple heuristic: if sigma is large relative to (f_best - mu),
    # boost the score to encourage exploration.
    
    # Alternative: use the "Probability of Improvement" weighted by sigma
    # PI = norm.cdf(z)
    # PIS = PI * sigma (Probability of Improvement with scale)
    
    # Let's try: score = LogEI + lambda * sigma * PI
    # where lambda is a small constant
    
    # Check time
    if time.time() > deadline:
        scores = np.where(np.isfinite(log_ei), log_ei, 0.0)
        return scores
    
    # Try multiple strategies and pick the best one? No, we need to return a single score vector.
    # Let's try a fixed combination.
    
    # Strategy 1: Pure LogEI
    score1 = log_ei
    
    # Strategy 2: LogEI + small UCB component
    kappa = 0.5
    ucb = -mu + kappa * sigma_safe
    # Normalize UCB to have zero mean and unit variance for fair combination
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-8
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Scale UCB to have similar magnitude as LogEI
    log_ei_mean = np.mean(log_ei)
    log_ei_std = np.std(log_ei) + 1e-8
    log_ei_norm = (log_ei - log_ei_mean) / log_ei_std
    
    # Combine: score = log_ei_norm + weight * ucb_norm
    weight = 0.3
    score2 = log_ei_norm + weight * ucb_norm
    
    # Check time
    if time.time() > deadline:
        scores = np.where(np.isfinite(score2), score2, 0.0)
        return scores
    
    # Strategy 3: Use a "diversity" proxy based on sigma
    # Points with high sigma are more likely to be in unexplored regions
    # Boost high-sigma points slightly
    sigma_norm = (sigma_safe - np.mean(sigma_safe)) / (np.std(sigma_safe) + 1e-8)
    score3 = log_ei_norm + 0.2 * sigma_norm
    
    # Check time
    if time.time() > deadline:
        scores = np.where(np.isfinite(score3), score3, 0.0)
        return scores
    
    # Let's try a different approach: use the "Knowledge Gradient" approximation
    # KG is hard to compute without coordinates, so let's stick with simpler methods.
    
    # Another idea: use a "regret" based acquisition
    # Minimize expected regret = E[f(x) - f_best] = mu - f_best + (sigma * something)
    # This is essentially EI.
    
    # Let's try a "confidence bound" approach with adaptive kappa
    # kappa = log(1 + iteration) / log(1 + total_iterations)
    # But we don't know the iteration.
    
    # Let's just use a fixed combination that has worked well in practice.
    # Based on literature, a combination of EI and UCB with weight 0.2-0.5 often works well.
    
    # Use score2 (LogEI + 0.3 * UCB)
    scores = score2
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
