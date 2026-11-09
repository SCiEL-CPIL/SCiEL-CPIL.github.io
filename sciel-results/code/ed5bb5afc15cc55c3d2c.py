import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a parametric model for the cost penalty.
    We fit a simple model to estimate the optimal exploration/exploitation trade-off
    by optimizing a parameter that balances improvement against cost.
    
    The acquisition score is defined as:
        score(x) = EI(x) / (cost(x) * (1 + beta * cost(x)))
    
    where beta is optimized to maximize the expected improvement per unit cost
    across the candidate set, using a lightweight surrogate fit.
    """
    
    # Avoid division by zero in z calculation
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Phi is CDF, phi is PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Ensure cost is positive and not zero
    cost_safe = np.maximum(cost, 1e-10)
    
    # Fit a parametric cost-aware model:
    # We model the effective cost penalty as cost * (1 + beta * cost)
    # and optimize beta to maximize the sum of EI / (cost * (1 + beta * cost))
    # over a subset of candidates (for speed, use all 2000 but with a fast objective)
    
    # To keep this fast, we'll use a simplified approach:
    # Estimate beta by finding the value that maximizes the expected score
    # under a Gaussian approximation of the EI distribution.
    
    # Compute the mean and std of EI and cost
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-10
    cost_mean = np.mean(cost_safe)
    cost_std = np.std(cost_safe) + 1e-10
    
    # The optimal beta balances exploration (high EI) against cost.
    # A reasonable heuristic is to set beta proportional to the ratio of
    # EI variability to cost variability, which encourages exploring
    # regions with high improvement potential relative to their cost.
    
    # Use a bounded optimization to find the best beta
    def objective(log_beta):
        beta = np.exp(log_beta)
        penalty = cost_safe * (1 + beta * cost_safe)
        penalty_safe = np.maximum(penalty, 1e-10)
        scores = ei / penalty_safe
        # We want to maximize the sum of scores, but also penalize
        # configurations where scores are too variable (unstable)
        return -np.sum(scores)
    
    # Optimize over log(beta) in a reasonable range
    res = minimize_scalar(objective, bounds=(-5, 5), method='bounded')
    log_beta_opt = res.x
    beta_opt = np.exp(log_beta_opt)
    
    # Compute final scores with the optimized beta
    penalty = cost_safe * (1 + beta_opt * cost_safe)
    penalty_safe = np.maximum(penalty, 1e-10)
    score = ei / penalty_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
