import numpy as np
from scipy.stats import norm
import time

# Global variable to track start time for deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 285.0  # Safety margin before 300s

def acquisition(mu, sigma, f_best):
    # Deadline guard: If we are very close to the time limit,
    # return a score that strongly favors exploitation (low mu)
    # to avoid wasting time on exploration or complex calculations.
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to pure exploitation: lower mu is better
        # Invert mu so higher score = lower mu
        return -mu
    
    # Ensure sigma is safe for division
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Use Upper Confidence Bound (UCB) for better exploration-exploitation balance
    # UCB = -mu + beta * sigma (minimization problem)
    # The finding shows UCB with explicit balance outperforms pure EI in low-budget regime
    # We combine EI and UCB: use UCB as the primary scoring mechanism
    # beta is chosen to balance exploration; for 2D Ackley with [-5,5]^2, 
    # a moderate beta works well. We use a time-adaptive beta that starts
    # with more exploration and decreases over time.
    
    # Estimate progress: if f_best is very small, we're close to optimum,
    # so reduce exploration. Otherwise, maintain exploration.
    # For Ackley, the minimum is 0, so f_best approaching 0 means we're done.
    # Use a simple heuristic: beta decreases as f_best gets closer to 0.
    # At start, f_best might be ~2-3, at end it might be ~0.001.
    # beta = 1.0 * (1.0 + f_best) / (1.0 + f_best) ... let's just use a fixed reasonable beta.
    # Based on the measured improvement (1.0 -> 1.538), UCB works well.
    # Let's use beta = 1.0 as a starting point, which gives UCB = -mu + sigma.
    
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    
    # Combine: use UCB as the primary score, but add a small EI component
    # to ensure we still have some exploitation pressure from the improvement-based term.
    # Actually, the finding says "Replace EI with UCB", so let's just use UCB.
    # But let's be safe and blend: score = 0.5 * ei + 0.5 * ucb_normalized
    # Actually, simpler: just use UCB since it measured 1.538.
    
    score = ucb
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
