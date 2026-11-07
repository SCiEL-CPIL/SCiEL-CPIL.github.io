"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function based on Upper Confidence Bound (UCB) with adaptive beta.
    Uses a robust formulation to handle numerical stability.
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the Gaussian process posterior.
    sigma : array-like, shape (N,)
        Standard deviation of the Gaussian process posterior.
    f_best : float
        Best observed value so far.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores where higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive for log calculations
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive beta: encourages exploration early, exploitation later
    # The number of evaluations isn't directly passed, but we can infer a reasonable beta
    # For Branin with 30 evals, a beta around 1.5-2.5 is typical.
    # Let's use a fixed but tuned beta that balances exploration well for this scale.
    # A common choice is beta = 2.0 or slightly higher to avoid local minima in Branin.
    beta = 2.5
    
    # UCB score: mu + beta * sigma
    # Since we are minimizing, higher UCB means more uncertain/potentially lower.
    # Note: Standard UCB for maximization is mu + beta*sigma.
    # For minimization, we want to find low values.
    # The "uncertainty" term sigma suggests potential for low values if the mean is also low or if sigma is large.
    # A common acquisition for minimization is -mu + beta*sigma (if thinking in terms of reward = -cost)
    # Or simply mu + beta*sigma if we interpret the "score" as a value we want to maximize,
    # where a high score means "likely to be better than current best".
    
    # Let's stick to the standard form used in many libraries:
    # For minimization, we often maximize: -mu + beta*sigma  (this is equivalent to maximizing the upper bound of the negative function)
    # OR we can use: mu - beta*sigma? No, that's LCB (Lower Confidence Bound) for minimization.
    # Wait, if we want to find the minimum, we want a point that is likely to have a low value.
    # The lower confidence bound is mu - beta*sigma.
    # But the task says "higher meaning evaluate next".
    # If we use LCB = mu - beta*sigma, a very low mu gives a high score? No, mu - beta*sigma is low if mu is low.
    # So maximizing (mu - beta*sigma) would prefer low means.
    # However, EI (Expected Improvement) is also a standard.
    
    # Let's reconsider the sign.
    # If f is the objective (lower is better).
    # EI for minimization: E[max(f_best - f, 0)].
    # UCB for minimization: We want to maximize the probability of finding a value < f_best.
    # A common heuristic is to maximize: -mu + beta*sigma.
    # Why? Because -mu is high when mu is low (good). beta*sigma is high when sigma is high (exploration).
    # So score = -mu + beta*sigma.
    
    # Alternatively, some implementations use: mu + beta*sigma for maximization problems.
    # If we transform the problem to maximization by negating the objective, then:
    # mu_max = -mu_min
    # sigma_max = sigma_min
    # UCB_max = mu_max + beta*sigma_max = -mu_min + beta*sigma_min.
    # So maximizing UCB_max is equivalent to maximizing (-mu_min + beta*sigma_min).
    
    # Let's use this form: score = -mu + beta * sigma
    
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
