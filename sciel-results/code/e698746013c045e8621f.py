import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    UCB balances exploration and exploitation by adding a term proportional to
    the standard deviation (uncertainty) to the negative mean (exploitation).
    For the multimodal Weierstrass function, a higher exploration coefficient
    helps escape local minima during the limited evaluation budget.
    
    Args:
        mu: Array of shape (2000,) with posterior means.
        sigma: Array of shape (2000,) with posterior standard deviations.
        f_best: The lowest observed objective value so far.
        
    Returns:
        Array of shape (2000,) with acquisition scores. Higher scores are preferred.
    """
    # Exploration coefficient. 
    # 1.5 was measured to improve over EI (score 1.0 -> 1.52).
    # We increase it to 2.0 to encourage more exploration in the multimodal landscape.
    beta = 2.0
    
    # Ensure sigma is strictly positive to avoid any numerical edge cases, 
    # though UCB does not involve division by sigma.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate UCB scores: -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    return scores
