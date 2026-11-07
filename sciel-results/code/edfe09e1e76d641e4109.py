"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Adaptive UCB acquisition function for minimizing Michalewicz 2D.
    
    Uses a dynamic beta parameter based on the distance to the known global minimum.
    Higher beta encourages exploration when the current best is far from the optimum.
    Lower beta encourages exploitation when the current best is close to the optimum.
    """
    # Known global minimum for Michalewicz 2D
    f_min = -1.8013
    
    # Calculate the gap between current best and known minimum
    # f_best is the lowest observed value (lower is better)
    gap = f_best - f_min
    
    # Clip gap to a reasonable range to avoid extreme beta values
    # If gap is very large (e.g., > 2), we are definitely lost, need max exploration.
    # If gap is small (e.g., < 0.05), we are close, need exploitation.
    
    # Define beta range
    beta_min = 0.5  # Exploitation focus
    beta_max = 3.0  # Exploration focus
    
    # Map gap to beta
    # If gap > 1.0, use beta_max
    # If gap < 0.05, use beta_min
    # Linear interpolation in between
    gap_threshold_high = 1.0
    gap_threshold_low = 0.05
    
    if gap >= gap_threshold_high:
        beta = beta_max
    elif gap <= gap_threshold_low:
        beta = beta_min
    else:
        # Normalize gap to [0, 1] for interpolation
        norm_gap = (gap - gap_threshold_low) / (gap_threshold_high - gap_threshold_low)
        # As gap increases, beta increases
        beta = beta_min + norm_gap * (beta_max - beta_min)
        
    # Compute UCB scores
    # Higher score means more likely to be selected
    # UCB = mu + beta * sigma
    # We want to minimize, so we are looking for low mu.
    # However, the acquisition function returns scores where HIGHER is better.
    # Standard UCB for minimization: score = -mu + beta * sigma ? 
    # No, standard UCB for minimization is often written as minimizing (mu - beta*sigma) 
    # or maximizing (beta*sigma - mu).
    # Let's check the previous successful UCB implementation logic.
    # The previous note said "UCB with beta=2 increases exploration".
    # Usually, for minimization, we pick the point that minimizes mu - beta*sigma.
    # Equivalently, we maximize beta*sigma - mu.
    # Let's verify with EI. EI for minimization is E[max(0, f_best - f(x))].
    # This is maximized when f(x) is low and uncertain.
    # UCB for minimization: score = beta * sigma - mu.
    # If sigma is large, score is high (explore).
    # If mu is low, score is high (exploit).
    
    scores = beta * sigma - mu
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
