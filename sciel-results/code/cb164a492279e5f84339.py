import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using UCB (Upper Confidence Bound) for minimization.
    
    The task is to minimize f(x). The standard UCB for maximization is mu + kappa*sigma.
    For minimization, we want to maximize the "potential for improvement".
    Since f_best is the lowest observed value, we are looking for points where the 
    lower bound of the prediction is low.
    
    A common approach for cost-aware BO is to normalize the acquisition by cost.
    Here we use a UCB-like score: (f_best - mu + kappa * sigma) / cost.
    This favors points with low predicted mean (exploitation) and high uncertainty (exploration).
    
    We use a fixed kappa that provides a good balance for the 2000-candidate screening.
    The cost is in [1, 2]. Dividing by cost penalizes expensive evaluations.
    
    Note: The previous program used EI/cost. EI is local. UCB is global.
    Given the Ackley function's many local minima, a global exploration strategy 
    like UCB might find the global minimum faster than a greedy EI approach 
    which can get stuck in local optima if the GP doesn't capture the global structure well.
    """
    # Ensure sigma is non-negative and avoid division by zero in any implicit operations
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate UCB for minimization.
    # We want to maximize the score.
    # The "improvement" potential is roughly (f_best - mu).
    # Adding kappa * sigma encourages exploration.
    # kappa is chosen to be around 2.0-3.0 for standard UCB. 
    # Since we are screening 2000 points, a higher kappa might be needed to ensure 
    # that we don't just cluster around the current best. 
    # However, the GP is standardized, so sigma is in objective units.
    # Let's use kappa = 2.0.
    
    kappa = 2.0
    
    # Score = (f_best - mu + kappa * sigma) / cost
    # This is positive when mu < f_best + kappa*sigma, which is almost always true 
    # for points not already known to be terrible.
    # It effectively ranks points by their upper bound of performance (lower is better for f, 
    # so we look at the lower bound of f? No, UCB for minimization is usually 
    # mu - kappa*sigma for the value, but for acquisition we want to pick the point 
    # that minimizes the lower bound? 
    # Actually, standard UCB for minimization to find the global minimum:
    # We want to maximize the probability of finding a value < f_best.
    # This is equivalent to maximizing the "expected improvement" or using 
    # mu - kappa*sigma as a proxy for the best possible value?
    # Wait, if I use mu - kappa*sigma, I am picking the point with the lowest 
    # lower-bound. This is a very aggressive exploitation/exploration mix.
    
    # Let's stick to the standard formulation for minimization acquisition:
    # We want to maximize the "upper confidence bound" of the *improvement*.
    # Improvement = f_best - f(x).
    # Expected Improvement is E[max(f_best - f(x), 0)].
    # UCB for improvement: (f_best - mu) + kappa * sigma.
    # This is exactly what the formula above represents.
    
    score = (f_best - mu + kappa * sigma) / cost
    
    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
