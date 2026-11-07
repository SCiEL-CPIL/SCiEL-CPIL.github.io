import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 320.0  # 320s to leave 10s safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    The strategy uses a UCB score: -(mu - kappa * sigma).
    For minimization, we want to maximize the "upper confidence bound" of the
    potential improvement, which corresponds to minimizing the lower bound of
    the function value given uncertainty. 
    Score = -(mu - kappa * sigma) = -mu + kappa * sigma.
    High sigma (uncertainty) increases the score, encouraging exploration.
    Low mu (good value) increases the score (via -mu), encouraging exploitation.
    
    We use a decaying kappa to shift from exploration to exploitation as the
    number of evaluations increases. Since we don't have access to the exact
    step count directly, we can approximate progress using the relationship
    between f_best and the initial scale, or simply use a fixed high kappa 
    that balances the known multimodality of Weierstrass.
    
    Given the constraint of 25 evaluations total (5 initial + 20 steps), 
    and the multimodal nature, a relatively high kappa is needed early on.
    However, without step info, a fixed kappa is safer. 
    Let's use a kappa that scales with the current uncertainty range if possible, 
    or a fixed value tuned for this specific problem.
    
    For Weierstrass 1D on [-0.5, 0.5], the function has many local minima.
    EI (current) is greedy. UCB with kappa ~ 2-5 is often better for multimodal.
    Let's try a fixed kappa = 3.0. This is a structural change from EI's 
    dynamic z-score weighting to a linear combination of mean and std.
    
    Parameters:
    mu (np.ndarray): Posterior mean at candidate points.
    sigma (np.ndarray): Posterior standard deviation at candidate points.
    f_best (float): Best observed function value so far.
    
    Returns:
    np.ndarray: Scores for each candidate point. Higher score is better.
    """
    # Deadline guard: if we're too close to the limit, use a fast deterministic fallback
    if time.time() > _DEADLINE - 5.0:
        # Fallback: pure exploitation (negate mean) or very small exploration
        return -mu
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB for minimization:
    # We want to find the point with the lowest possible value.
    # The "lower confidence bound" (LCB) for the function value is mu - kappa * sigma.
    # We want to MINIMIZE this LCB.
    # The acquisition function should return a score where HIGHER is better.
    # So, score = -LCB = -(mu - kappa * sigma) = -mu + kappa * sigma.
    
    # Choose kappa. 
    # In standard BO, kappa is often tuned. For multimodal functions like Weierstrass,
    # higher kappa helps escape local optima.
    # Let's use a fixed kappa. 2.0 to 5.0 is a common range.
    # Let's try 3.0.
    kappa = 3.0
    
    scores = -mu + kappa * sigma
    
    return scores
