# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    LCB encourages exploration of regions with high uncertainty (sigma) 
    even if the mean (mu) is not yet below f_best, which is crucial for 
    multimodal functions like Rastrigin.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # LCB parameter: higher beta encourages more exploration
    # For Rastrigin, which has many local minima, we need significant exploration
    beta = 2.0

    # Lower Confidence Bound: mu - beta * sigma
    # We want to minimize this, so for an acquisition function (maximize score),
    # we use the negative of LCB or just LCB if we define score as "desirability".
    # Standard LCB for minimization: score = mu - beta * sigma.
    # Higher score means "better" (lower predicted value).
    lcb = mu - beta * sigma_safe

    # Cost-aware LCB: divide by cost to prioritize cheaper evaluations
    # This aligns with the "cost-aware" requirement
    scores = lcb / cost_safe

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
