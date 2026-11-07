# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a Bayesian Regret Minimization (BRM) inspired approach:
    - Compute the probability that each candidate is the global minimum
      within the remaining budget, weighted by expected improvement.
    - Normalize by cost to account for evaluation expense.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x) scaled to [0,1]).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative sigma
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # BRM-inspired weighting:
    # The probability that candidate i is the global minimum within budget
    # is approximated by the probability that mu_i is the lowest among all candidates,
    # weighted by the uncertainty. We use a softmax-like weighting based on
    # the expected minimum value, which favors candidates with low mu and moderate sigma.
    
    # Approximate the probability of being the global minimum by considering
    # the expected minimum of the GP posterior. We use a two-stage approach:
    # 1. Compute a "regret potential" score for each candidate
    # 2. Weight by the probability of being among the best candidates
    
    # Compute the expected minimum contribution:
    # For each candidate, the expected contribution to the minimum is
    # approximately mu_i - 0.5 * sigma_i^2 / (f_best - mu_i + epsilon) when mu_i < f_best
    # This is a simplification of the Bayesian regret minimization objective.
    
    # A more principled approach: use the fact that the probability that
    # candidate i achieves a value below f_best is norm.cdf(z_i), and the
    # expected improvement is EI_i. The BRM objective weights EI by the
    # probability that no other candidate will do better.
    
    # Approximation: weight EI by exp(-max(0, f_best - mu_i) / sigma_i^2)
    # which downweights candidates that are already well below f_best but
    # have high uncertainty (they might not be as good as they seem).
    
    # Actually, let's use a cleaner BRM-inspired formulation:
    # The Bayesian regret is E[min_x f(x) - f(x*)] where x* is the true minimizer.
    # Under the GP posterior, the expected minimum is approximately:
    # E[min_i f_i] ≈ min_i (mu_i - alpha * sigma_i) for some alpha
    # 
    # We approximate the acquisition as:
    # score_i = EI_i / cost_i * w_i
    # where w_i captures the probability that candidate i is the best within budget.
    
    # Compute a softmax weighting based on the "quality" of each candidate:
    # quality_i = -(mu_i) - beta * sigma_i (penalize high uncertainty slightly)
    # Use a temperature-scaled softmax to get probabilities
    
    # Temperature based on the spread of mu values
    mu_range = mu.max() - mu.min()
    if mu_range < 1e-6:
        temp = 1.0
    else:
        temp = mu_range / 5.0  # Spread the probabilities
    
    # Quality score: lower is better, so we negate
    quality = -mu / temp
    # Subtract max for numerical stability
    quality_shifted = quality - quality.max()
    weights = np.exp(quality_shifted)
    weights = weights / (weights.sum() + 1e-12)
    
    # Combine EI with the probability weighting
    # The BRM-inspired score is:
    # score_i = (EI_i * weights_i) / cost_i
    # This favors candidates with high expected improvement that are also
    # likely to be among the best candidates (high weight).
    
    scores = (ei * weights) / cost
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
