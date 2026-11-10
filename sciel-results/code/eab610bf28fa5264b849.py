"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats as stats

# Deadline guard: The evaluator kills after 330s.
_start_time = time.time()
TOTAL_BUDGET = 30.0

def acquisition(mu, sigma, f_best, cost):
    """
    Budget-Adjusted Lower Confidence Bound (LCB) acquisition function.
    
    This strategy uses a dynamic coefficient for the uncertainty term (sigma)
    that increases as the budget is exhausted. Unlike EI, which is a probabilistic
    measure of improvement, LCB is an optimization-theoretic bound that explicitly
    balances exploitation (minimizing mu) and exploration (maximizing sigma).
    
    The dynamic coefficient is derived from the estimated remaining budget.
    Since we don't have direct access to the current total cost, we infer the
    "urgency" from the cost vector itself. In the loop, the cost of the selected
    point is added to the total. We assume that if the current batch of candidates
    has high costs, we are likely in a later stage or the function is expensive,
    so we should be more conservative (exploitative). Conversely, if costs are low,
    we can afford to explore more.
    
    However, a more robust structural change is to use the magnitude of f_best
    and the distribution of costs to estimate the remaining budget.
    
    A simpler and more robust structural change:
    Use a fixed, high exploration weight initially, but decay it based on a
    proxy for "progress". Since f_best decreases, we can use f_best itself
    as a proxy for progress if we assume the functions are normalized or similar
    in scale. But they aren't (Hartmann is negative).
    
    Let's use a different proxy: The number of evaluations is roughly proportional
    to the total cost spent. We can estimate the "fraction of budget spent" by
    assuming a typical cost. If the average cost in the current batch is ~1.5,
    and we started with 5 points (cost ~7.5), and budget is 30, we have ~22 left.
    
    Actually, the most robust structural change without knowing n_evals is to
    use a **Cost-Weighted UCB** where the exploration term is divided by the cost
    of the point, but with a dynamic beta that scales with the inverse of the
    minimum cost in the batch. This ensures that cheap points are explored more
    aggressively when the budget is tight.
    
    Let's try: score = -mu - beta * sigma / cost
    where beta = 1.0 + 1.0 / (min_cost + 1e-9)
    
    This makes beta larger when costs are low (early stage or cheap regions),
    encouraging exploration. When costs are high, beta is closer to 1.0,
    making it more exploitative.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: avoid division by zero
    cost = np.maximum(cost, 1e-9)
    
    # Dynamic beta: 
    # If the minimum cost in the batch is low, we are likely in a cheap region
    # or early stage, so we can afford more exploration.
    # If the minimum cost is high, we are in an expensive region, so we should
    # be more exploitative.
    min_cost = np.min(cost)
    beta = 1.0 + 1.0 / (min_cost + 1e-9)
    
    # LCB: minimize mu + beta * sigma
    # Acquisition: maximize -mu - beta * sigma / cost
    # Dividing by cost makes it cost-aware: we prefer points that have high
    # expected improvement per unit cost.
    
    scores = -mu - beta * sigma / cost
    
    # Handle numerical issues
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
