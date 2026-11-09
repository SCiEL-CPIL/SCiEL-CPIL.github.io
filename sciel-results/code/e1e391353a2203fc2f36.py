import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior
    f_best : float
        Best observed function value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean of scaled coordinates)
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip small negative values due to numerical precision
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    base_score = ei / cost
    
    # Bounded UCB term to balance exploration and exploitation
    # UCB = mu + kappa * sigma
    # We use a bounded version that is normalized by cost
    # The UCB term encourages exploration of high-uncertainty regions
    # while being bounded to prevent excessive exploration
    kappa = 1.0  # Exploration parameter
    ucb = mu + kappa * sigma_safe
    
    # Normalize UCB by cost to make it cost-aware
    # We subtract f_best to make it a "regret-like" quantity
    # and clip to ensure it's non-negative (we only care about improvement)
    ucb_score = np.maximum(f_best - ucb, 0.0) / cost
    
    # Combine EI and UCB terms
    # The UCB term is weighted to be smaller than the EI term
    # to prioritize exploitation while still allowing exploration
    combined_score = base_score + 0.3 * ucb_score
    
    # Prior bias: The minimizer is in the inner 80% of [-4, 5]^4.
    # In scaled coordinates [0,1], the inner 80% region is [0.1, 0.9].
    # The center of this region is 0.5, so expected minimizer location is 0.5.
    # Expected cost at minimizer = 1 + 0.5 = 1.5.
    target_cost = 1.5
    
    # Gaussian weight on cost deviation to focus search around expected cost
    # This biases towards candidates with cost close to 1.5
    lambda_bias = 1.5
    weight = np.exp(-lambda_bias * (cost - target_cost) ** 2)
    
    # Exploration bonus for high uncertainty
    # When EI is low but sigma is high, we want to explore more
    # This helps avoid premature convergence
    exploration_bonus = sigma_safe / cost
    
    # Combine base score and exploration bonus
    # The exploration bonus is weighted to be smaller than the base score
    score = (combined_score + 0.15 * exploration_bonus) * weight
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
