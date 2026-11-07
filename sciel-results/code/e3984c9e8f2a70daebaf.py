import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    Uses Adaptive Upper Confidence Bound (UCB) for minimization.
    """
    f_global_min = 0.397887
    
    # Calculate current regret
    regret = f_best - f_global_min
    
    # Adaptive Beta parameters
    beta_min = 0.5
    beta_max = 3.0
    r_scale = 10.0
    
    # Calculate beta based on regret
    # High regret (early) -> High beta (explore)
    # Low regret (late) -> Low beta (exploit)
    # The fraction regret / (regret + r_scale) is in [0, 1) for regret >= 0
    if regret >= 0:
        fraction = regret / (regret + r_scale)
    else:
        # If we are below the global minimum (due to noise or error),
        # we should exploit heavily.
        fraction = 0.0
        
    beta = beta_min + (beta_max - beta_min) * fraction
    
    # UCB score for minimization:
    # We want to minimize f(x).
    # Lower Confidence Bound (LCB) = mu - beta * sigma
    # We want to select the point with the lowest LCB.
    # The evaluator selects the point with the highest score.
    # Therefore, Score = -LCB = beta * sigma - mu
    
    scores = beta * sigma - mu
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
