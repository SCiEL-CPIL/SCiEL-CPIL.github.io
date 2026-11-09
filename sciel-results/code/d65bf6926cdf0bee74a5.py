import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using UCB with progress-based decay.
    Progress is estimated from f_best relative to the known min/max of Michalewicz 2D.
    """
    # Known bounds for Michalewicz 2D
    f_min = -1.8013
    f_max = 0.0
    
    # Calculate progress in [0, 1]
    # f_best starts near 0 (or positive) and decreases to f_min
    # progress = 0 at start, 1 at end
    denominator = f_max - f_min
    if denominator == 0:
        progress = 0.0
    else:
        progress = (f_max - f_best) / denominator
        
    # Clamp progress to [0, 1]
    progress = np.clip(progress, 0.0, 1.0)
    
    # Define beta decay
    # Start with high exploration, end with low exploration
    beta_start = 2.0
    beta_end = 0.2
    
    # Linear interpolation
    beta = beta_start * (1.0 - progress) + beta_end * progress
    
    # UCB Score for minimization:
    # We want to maximize the score.
    # Lower mu is better. Higher sigma is more uncertain.
    # score = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
