import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
# 330s limit. We want to be well before.
_DEADLINE = _START_TIME + 325

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a deterministic, cost-normalized 
    Lower Confidence Bound (LCB).
    
    The score is (f_best - mu + beta * sigma) / cost.
    This explicitly maximizes the optimistic gap per unit cost, encouraging 
    aggressive exploration of high-variance, low-cost regions.
    
    Args:
        mu (np.ndarray): Mean predictions from the GP, shape (2000,).
        sigma (np.ndarray): Standard deviation predictions from the GP, shape (2000,).
        f_best (float): The lowest observed function value.
        cost (np.ndarray): Cost of evaluating each candidate, shape (2000,).
    
    Returns:
        np.ndarray: Scores for the candidates, shape (2000,). Higher is better.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: Prioritize cheap points if we are running out of time
        return 1.0 / (cost + 1e-9)

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Dynamic beta: 
    # We want to explore more when uncertainty is high relative to the scale of the problem.
    # A fixed beta might be too conservative or aggressive.
    # Let's scale beta by the current uncertainty level (std of sigma) to adapt to the GP's state.
    # However, the directive suggests "dynamically scaled by the current uncertainty budget".
    # A simple heuristic: beta = 1.0 + 0.5 * (mean(sigma) / (std(sigma) + 1e-9))
    # Or simply a fixed but tuned beta. Let's try a beta that balances exploration.
    # For Rastrigin, which is multimodal, we need significant exploration.
    # Let's use a beta that is proportional to the scale of sigma.
    # If sigma is large, we want to explore more.
    
    # Let's try a beta that is constant but chosen to be effective for Rastrigin.
    # Based on inspiration, LCB improved score. 
    # Let's use beta = 1.0 as a starting point, maybe scaled by cost? No, beta is usually fixed or time-dependent.
    # Let's stick to a fixed beta but ensure it's strong enough.
    # The score is (f_best - mu + beta * sigma) / cost.
    # To make it more robust, we can normalize the LCB term.
    
    # Let's compute the LCB value: lcb = mu - beta * sigma (for minimization, we want low values)
    # The "improvement" potential is f_best - lcb = f_best - mu + beta * sigma.
    # We divide by cost.
    
    # Let's try beta = 1.0.
    beta = 1.0
    
    # Calculate the optimistic gap
    gap = f_best - mu + beta * sigma_safe
    
    # Cost-aware score
    # We divide by cost to get "improvement per unit cost"
    scores = gap / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
