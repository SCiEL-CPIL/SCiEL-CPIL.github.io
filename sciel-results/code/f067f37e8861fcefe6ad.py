"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    to balance exploration and exploitation. The blend is adaptive based on
    the current uncertainty relative to the best known value.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Upper Confidence Bound (UCB) for minimization
    # UCB = mu - beta * sigma
    # We need to choose beta appropriately. A common choice is beta = 2 or adaptive.
    # Since we don't have the step number, we'll use a fixed beta that balances exploration.
    # In early stages, we want more exploration; later, more exploitation.
    # However, without step info, we can use a moderate beta.
    # Let's use beta = 1.5 as a reasonable default for 1D problems.
    beta = 1.5
    ucb = mu - beta * sigma_safe
    
    # We need to combine EI and UCB. Since they have different scales, we should normalize them.
    # Normalize EI: EI is non-negative, higher is better.
    # Normalize UCB: UCB is in objective units, lower is better for minimization.
    # To combine, we can convert UCB to a "score" where higher is better.
    # ucb_score = -ucb (since lower UCB means more promising for minimization)
    # But we need to make them comparable.
    
    # Alternative approach: Use a weighted sum after normalizing both to similar scales.
    # Let's normalize EI to [0, 1] or similar range.
    # Actually, a simpler and effective approach is to use a hybrid formula:
    # score = w * EI + (1 - w) * (f_best - mu - beta * sigma)
    # But this doesn't quite work because EI and UCB have different interpretations.
    
    # Better approach: Use EI as the primary acquisition and add an exploration bonus.
    # Or use a well-known hybrid: qEI or similar.
    
    # Let's try a different strategy: Use EI but with an adaptive temperature or
    # add a small exploration term.
    
    # Actually, the most robust approach for 1D Bayesian optimization is often
    # to use EI with a small exploration bonus based on the posterior variance.
    
    # Let's use: score = EI + gamma * sigma
    # where gamma controls the exploration-exploitation trade-off.
    # gamma should be small to not override EI too much.
    
    gamma = 0.1
    scores = ei + gamma * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
