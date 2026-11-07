# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Bayesian Regret Minimization (BRM) inspired acquisition function.
    
    Uses a multi-fidelity surrogate-guided ranking that approximates the expected
    minimum of the function conditioned on the cost budget by weighting the GP
    mean by the probability of being the global minimum within the remaining budget.
    
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
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-scores for expected improvement
    z = (f_best - mu) / sigma
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # BRM-inspired approach: approximate the probability that each candidate
    # is the global minimum within the remaining budget.
    
    # The probability that candidate i is the global minimum is related to
    # the probability that its value is lower than all other candidates.
    # Under the GP posterior, we can approximate this using the relative
    # ranking of the expected values and uncertainties.
    
    # Step 1: Compute a "regret potential" score for each candidate.
    # This combines the expected improvement with a measure of how likely
    # the candidate is to be the best within the budget.
    
    # For BRM, we want to minimize E[f(x) - f*] where f* is the true minimum.
    # A practical approximation is to weight EI by the probability that the
    # candidate achieves a value below f_best, which is norm.cdf(z_i).
    
    prob_below = norm.cdf(z)
    
    # Step 2: Compute a budget-aware weight that accounts for the cost.
    # Candidates with lower cost are more valuable because we can evaluate
    # more of them within the remaining budget.
    
    # Normalize cost to [0, 1] range for better conditioning
    cost_min = cost.min()
    cost_max = cost.max()
    if cost_max - cost_min > 1e-9:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)
    
    # Budget-aware weight: favor low-cost candidates
    # Use an exponential decay to softly penalize high-cost candidates
    budget_weight = np.exp(-2.0 * cost_norm)
    
    # Step 3: Compute a "global minimum probability" approximation.
    # Under the GP posterior, the probability that candidate i is the minimum
    # is approximately proportional to exp(-mu_i / tau) * exp(-sigma_i^2 / (2 * tau^2))
    # where tau is a temperature parameter related to the uncertainty scale.
    
    # Estimate the temperature from the data
    mu_range = mu.max() - mu.min()
    sigma_mean = sigma.mean()
    
    if mu_range > 1e-6:
        tau_mu = mu_range / 3.0
    else:
        tau_mu = 1.0
    
    if sigma_mean > 1e-6:
        tau_sigma = sigma_mean / 2.0
    else:
        tau_sigma = 1.0
    
    # Compute the log-probability of being the global minimum
    log_prob_min = -mu / tau_mu - (sigma ** 2) / (2 * tau_sigma ** 2)
    
    # Normalize to get probabilities (softmax)
    log_prob_min_shifted = log_prob_min - log_prob_min.max()
    prob_min = np.exp(log_prob_min_shifted)
    prob_min_sum = prob_min.sum()
    if prob_min_sum > 1e-12:
        prob_min = prob_min / prob_min_sum
    else:
        prob_min = np.ones_like(mu) / len(mu)
    
    # Step 4: Combine all components into a BRM-inspired score.
    # The score should reflect:
    # 1. Expected improvement (EI)
    # 2. Probability of being the global minimum within budget
    # 3. Cost efficiency (budget-aware weight)
    
    # The BRM objective can be approximated as:
    # score_i = EI_i * prob_min_i * budget_weight_i / cost_i
    
    scores = ei * prob_min * budget_weight / cost
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
