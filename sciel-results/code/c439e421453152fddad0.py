"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) for minimization.
    
    Args:
        mu: Mean of the Gaussian Process posterior (shape: 2000,).
        sigma: Standard deviation of the GP posterior (shape: 2000,).
        f_best: Best (lowest) function value observed so far.
        
    Returns:
        scores: Array of 2000 scores. Higher score indicates a better candidate.
    """
    # Ensure sigma is non-negative to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Fixed exploration constant for LCB.
    # A higher k encourages exploration of uncertain regions, which is 
    # beneficial for multimodal functions like Michalewicz where the global 
    # minimum might be in a region not yet sampled.
    k = 4.0
    
    # LCB for minimization: Q(x) = mu(x) - k * sigma(x)
    # We want to select the point with the *lowest* Q(x).
    # The acquisition score to be *maximized* is therefore -Q(x) = -mu(x) + k * sigma(x).
    scores = -mu + k * sigma
    
    # Replace any non-finite values with -inf to ensure they are never selected
    # and to satisfy the "finite scores" requirement (though -inf is not finite, 
    # the task says "Return 2000 finite scores". 
    # Wait, "anything else makes the whole result invalid". 
    # -inf is not finite. I must return finite numbers.
    # If a score is inf, it's likely due to numerical overflow.
    # We should clip to a large finite number or handle carefully.
    # However, mu and sigma from a GP should be finite.
    # If they are not, the GP failed.
    # To be safe, replace non-finite with a very low score (e.g., -1e10) 
    # so they are not picked, but remain finite.
    
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
