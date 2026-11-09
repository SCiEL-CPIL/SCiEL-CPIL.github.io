"""Rewritten program using a cost-aware Lower Confidence Bound (LCB) acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    For minimization, we want low mu and high uncertainty (sigma).
    The standard LCB is: mu - beta * sigma.
    To make it cost-aware, we penalize high costs. 
    A common heuristic is to divide by cost or subtract a penalty term.
    
    Let's try: (mu - beta * sigma) / cost.
    However, mu is in objective units. f_best is the current best.
    It might be more stable to normalize by the scale of the problem or use a relative measure.
    
    Alternative: Expected Improvement (EI) per unit cost is the current approach.
    Let's try a different family: Probability of Improvement (POI) per unit cost, or UCB-like but for minimization.
    
    Actually, let's look at the structure.
    mu: predicted mean
    sigma: predicted std
    f_best: best observed value
    cost: cost of evaluating each point
    
    We want to return scores where higher is better.
    
    Approach: Cost-aware LCB.
    LCB(x) = mu(x) - beta * sigma(x)
    Score(x) = -LCB(x) / cost(x)  ? No, we want higher score for better candidates.
    So Score(x) = ( -mu(x) + beta * sigma(x) ) / cost(x)
    
    We need to choose beta. A common adaptive beta is beta = 2 * log(1 + n) or similar, but we don't have n here directly, though we can infer it from cost?
    Total cost < 30. Initial 5 points cost ~ 1 + 0.5 = 1.5 each? No, cost = 1 + mean(x_scaled).
    Uniform in [-1, 1] scaled to [0,1] is uniform in [0,1]. Mean is 0.5. So initial cost per point is 1.5.
    5 points -> cost 7.5.
    Remaining budget 22.5.
    Number of additional evaluations approx 22.5 / 1.5 = 15.
    Total evals ~ 20.
    
    Let's use a fixed beta or a simple adaptive one.
    Since we don't have access to the number of observations directly in the function signature (only mu, sigma, f_best, cost), we can't easily compute an adaptive beta based on n.
    We can assume a moderate beta, e.g., 2.0 or 3.0.
    
    Let's try:
    beta = 2.0
    lcb = mu - beta * sigma
    score = -lcb / cost
    
    Wait, if mu is very low (good), -lcb is high.
    If sigma is high, -lcb is higher (explore).
    If cost is high, score is lower (exploit cheap points).
    
    This seems reasonable.
    
    Another option: Upper Confidence Bound for Minimization (UCB-min).
    UCB-min(x) = mu(x) - beta * sigma(x).
    We want to minimize UCB-min.
    So we want to maximize -UCB-min(x) / cost(x).
    
    Let's add a small epsilon to cost to avoid division by zero, though cost >= 1.
    
    We should also handle sigma being very small.
    
    Let's try this LCB approach.
    """
    beta = 2.0
    # Lower Confidence Bound for minimization
    lcb = mu - beta * sigma
    # We want to maximize the potential for improvement, so we look at how much lower it could be.
    # Score should be high when LCB is low (good) and cost is low.
    # Score = -LCB / cost
    scores = -lcb / cost
    return scores
# EVOLVE-BLOCK-END
