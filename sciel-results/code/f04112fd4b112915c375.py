"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement per unit cost, with a
    dynamic exploration bonus that decays as the budget is consumed.
    
    The cost of evaluating x is 1 + mean(x scaled to [0,1]).
    We estimate the cost for each candidate from the `cost` array.
    
    mu: (2000,) predicted mean
    sigma: (2000,) predicted std
    f_best: scalar, best observed value
    cost: (2000,) estimated cost of evaluating each candidate
    
    Returns: (2000,) acquisition scores (higher = better)
    """
    # Ensure finite inputs
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: EI per unit cost
    # cost is the evaluation cost for each candidate
    cost_safe = np.maximum(cost, 1e-9)
    cost_score = ei / cost_safe
    
    # Dynamic exploration: add a term that encourages exploration early
    # and exploitation later. The budget is 30 total cost.
    # We don't know the current total cost directly, but we can infer
    # from the cost array and the fact that we're being called repeatedly.
    # 
    # A simpler structural change: use a UCB-style term with a
    # budget-aware beta. Since we don't have direct access to iteration
    # count or remaining budget, we use the mean cost as a proxy for
    # how "expensive" the current candidates are, and the spread of mu
    # to gauge convergence.
    
    # Alternative: pure cost-normalized EI with a small UCB component
    # The UCB term uses sigma scaled by a factor that depends on the
    # relative position of f_best vs the mean prediction.
    
    # Let's use a hybrid: 80% cost-normalized EI + 20% UCB
    # UCB: mu + beta * sigma, where beta adapts
    # Since we're minimizing, we want low mu and we like uncertainty.
    # UCB for minimization: -mu + beta * sigma
    
    # Estimate remaining budget fraction from cost distribution
    # If cost is around 1.5 on average, and budget is 30, we've used
    # some fraction. We can't know exactly, so use a fixed blend.
    
    # Structural change: use Thompson Sampling-style score
    # Sample from the GP posterior and pick the best
    # But we need to return 2000 scores, not a single point.
    
    # Better structural change: Use a cost-aware UCB with adaptive beta
    # beta = base_beta * (1 + f_best / (abs(f_best) + 1))
    # This makes beta larger when f_best is large (early) and smaller when f_best is small (late)
    
    # For minimization, acquisition = -mu + beta * sigma - gamma * cost
    # But we want higher = better, so:
    # score = (f_best - mu) / cost + beta * sigma / cost
    
    # Let's try: cost-normalized EI with an additive UCB term
    # The UCB term encourages exploration in high-uncertainty regions
    # scaled by the inverse cost to prefer cheap exploratory points
    
    # Compute a simple beta based on the scale of the problem
    # f_best is in objective units; sigma is also in objective units
    # Use a relative beta
    
    # Structural: use a different acquisition family - Knowledge Gradient
    # KG approximates the expected reduction in the minimum
    # For simplicity, use a cost-normalized combination of EI and PI
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Blend EI and PI, both cost-normalized
    # EI emphasizes magnitude of improvement, PI emphasizes probability
    # Use EI primarily, with PI as a tiebreaker for close values
    
    # Final score: cost-normalized EI
    # Add a small UCB term to encourage exploration
    beta = 0.5  # fixed exploration-exploitation balance
    
    ucb_term = -mu + beta * sigma
    score = cost_score + 0.1 * ucb_term / cost_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
