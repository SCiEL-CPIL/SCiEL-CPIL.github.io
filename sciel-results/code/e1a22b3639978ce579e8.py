"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = 310.0


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Styblinski-Tang 2D.
    
    Structural change: Uses a cost-adjusted Lower Confidence Bound (LCB) which
    explicitly balances exploitation (low predicted value) and exploration
    (high uncertainty), weighted inversely by evaluation cost. This is more
    appropriate for the low-budget multimodal Styblinski-Tang problem than
    EI-based methods, as it avoids premature convergence to local minima by
    maintaining a principled exploration-exploitation tradeoff.
    
    The LCB for minimization is: LCB = mu - beta * sigma
    We want to maximize the acquisition score, so we use -LCB = -mu + beta*sigma.
    To incorporate cost, we divide by cost to penalize expensive evaluations.
    
    Additionally, we add a small Expected Improvement term to ensure we don't
    completely ignore known good regions, but weight it much lower than the
    exploration term to maintain broad search.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Lower Confidence Bound for minimization
    # We want to find points where the true value is likely low
    # LCB = mu - beta * sigma
    # Higher score = more likely to be good, so we use -LCB = -mu + beta*sigma
    # But we need to be careful: -mu can be very large positive if mu is very negative
    # Let's normalize or use a different formulation
    
    # Alternative: Use the negative of the LCB as the acquisition value
    # For minimization, we want to minimize f(x), so we want to find x where
    # the lower bound on f(x) is as low as possible
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to minimize LCB(x), which is equivalent to maximizing -LCB(x)
    # -LCB(x) = -mu(x) + beta * sigma(x)
    
    # Cost adjustment: divide by cost to penalize expensive evaluations
    # score = (-mu + beta * sigma) / cost
    
    # But -mu can have large magnitude. Let's think about this differently.
    # The acquisition function should be higher for points we want to evaluate.
    # For minimization with cost:
    # - Points with low mu (good predicted value) should score high
    # - Points with high sigma (uncertain) should score high (exploration)
    # - Points with low cost should score high (cheap)
    
    # A clean formulation: 
    # score = (f_best - mu + beta * sigma) / cost
    # This combines improvement potential (f_best - mu) with uncertainty (beta*sigma)
    # and penalizes by cost.
    
    # Actually, let's use a simpler and more standard approach:
    # Cost-adjusted UCB for minimization
    # UCB_min = mu - beta * sigma  (we want to minimize this)
    # Acquisition score = -UCB_min / cost = (-mu + beta * sigma) / cost
    
    # But we should also consider that we want to find values lower than f_best
    # So let's use: score = (f_best - mu + beta * sigma) / cost
    # This ensures that points with mu > f_best get lower scores (since f_best - mu < 0)
    # unless beta*sigma is large enough to compensate (exploration)
    
    # Choose beta: for low budget, we want more exploration
    # beta = 1.0 is a reasonable starting point for ~25 evaluations
    beta = 1.0
    
    # Compute the acquisition score
    score = (f_best - mu + beta * sigma_safe) / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Add a small expected improvement term to ensure we don't completely
    # ignore regions with high expected improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Weight EI much lower than the LCB term
    gamma = 0.1
    score = score + gamma * ei / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
