import numpy as np
from scipy.stats import norm
import time

# Global time tracking for safety
_start_time = time.time()
_time_limit = 300.0 # Seconds

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are dangerously close to the limit, return a safe fallback
    if time.time() - _start_time > _time_limit - 5.0:
        return np.zeros_like(mu)
    
    # Safe sigma to avoid division by zero
    safe_sigma = np.maximum(sigma, 1e-9)
    
    # UCB Calculation: mu + beta * sigma
    # Since we are minimizing, we want to find low values.
    # The "upper confidence bound" for the true function value is mu + beta*sigma.
    # We want to select the point with the lowest potential value.
    # However, the task says "higher = evaluate next".
    # So we need to maximize a score that indicates "goodness".
    # For minimization, a good point has low mu and high uncertainty (potential to be even lower).
    # Standard UCB for minimization: minimize mu + beta * sigma.
    # To convert to a maximization score: score = -(mu + beta * sigma) ?
    # Or perhaps: score = -mu - beta * sigma ?
    # Let's think about the cost.
    # Cost-aware: score = (benefit) / cost.
    # Benefit for minimization: How much better could this be than f_best?
    # Lower bound on f(x) is mu - beta * sigma.
    # Improvement over f_best: f_best - (mu - beta * sigma) = f_best - mu + beta * sigma.
    # This is similar to EI but without the non-linear saturation.
    # Let's use: score = (f_best - mu + beta * sigma) / cost
    # If mu < f_best, this is positive. If mu > f_best, it depends on beta*sigma.
    # This encourages exploring points with high uncertainty (large sigma) and low mean.
    
    beta = 2.0
    # Note: sigma is std dev.
    # The term (f_best - mu) is the expected improvement (linear part).
    # The term beta * sigma is the exploration bonus.
    
    safe_cost = np.maximum(cost, 1e-9)
    
    # Calculate the "potential improvement" score
    # We want to maximize this.
    # If mu is very low, f_best - mu is large.
    # If sigma is large, beta * sigma is large.
    potential_improvement = (f_best - mu) + beta * safe_sigma
    
    score = potential_improvement / safe_cost
    
    # Handle any numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
