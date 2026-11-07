# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using UCB (Upper Confidence Bound) for minimization.
    Score = (f_best - mu + kappa * sigma) / cost
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Finite acquisition scores (higher = more desirable).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive to avoid division by zero
    cost = np.maximum(cost, 1e-9)
    
    # Exploration-exploitation trade-off parameter
    # kappa controls how much we value uncertainty (exploration)
    # A value of 2.0 is a standard heuristic for balanced exploration.
    kappa = 2.0
    
    # Calculate the potential improvement using the Lower Confidence Bound (LCB)
    # for the function value, which corresponds to the Upper Confidence Bound (UCB)
    # for the improvement in a minimization context.
    # LCB(x) = mu(x) - kappa * sigma(x)
    # Potential Improvement = f_best - LCB(x) = f_best - mu(x) + kappa * sigma(x)
    improvement_potential = f_best - mu + kappa * sigma
    
    # We only consider positive improvements. If the potential improvement is negative,
    # it means the lower bound of the function value is still above the current best,
    # so it's unlikely to be an improvement. We set these to 0.
    improvement_potential = np.maximum(improvement_potential, 0.0)
    
    # Normalize by cost to prefer cheaper evaluations
    scores = improvement_potential / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
