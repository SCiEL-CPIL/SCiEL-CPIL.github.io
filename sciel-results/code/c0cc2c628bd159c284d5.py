"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s hard kill

def _ei(mu, sigma, f_best):
    """Compute Expected Improvement for minimization."""
    sigma = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

def _ucb(mu, sigma, beta):
    """Compute UCB for minimization: -mu + beta * sigma."""
    return -mu + beta * sigma

def _cost_aware_ucb(mu, sigma, f_best, cost):
    """
    Cost-aware UCB variant.
    Score = (f_best - mu) / cost + sigma / cost^2
    This is structurally different from standard EI:
    - Exploitation is linear in (f_best - mu) divided by cost
    - Exploration is sigma divided by cost squared (heavier penalty for expensive exploration)
    """
    term_exploit = (f_best - mu) / cost
    term_explore = sigma / (cost * cost)
    return term_exploit + term_explore

def _cost_aware_ei(mu, sigma, f_best, cost):
    """
    Standard EI divided by cost.
    """
    ei = _ei(mu, sigma, f_best)
    return ei / cost

def _hybrid_score(mu, sigma, f_best, cost, w_ucb):
    """
    Blend of cost-aware UCB and cost-aware EI.
    w_ucb controls the mix: 0.0 = pure EI/cost, 1.0 = pure UCB/cost
    """
    score_ei = _cost_aware_ei(mu, sigma, f_best, cost)
    score_ucb = _cost_aware_ucb(mu, sigma, f_best, cost)
    return (1.0 - w_ucb) * score_ei + w_ucb * score_ucb

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a structural blend of
    Cost-Aware EI and Cost-Aware UCB, with adaptive mixing based on
    optimization progress estimated from f_best.
    
    Parameters
 ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far (lower is better).
    cost : np.ndarray, shape (2000,)
        Evaluation cost for each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = better to evaluate next).
    """
    # Time guard
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        fallback = ei / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Estimate progress from f_best
    # For the given functions, the optimum is 0 (or -3.86 for Hartmann)
    # Early in optimization, f_best is large (bad); late, f_best is small (good)
    # Use a sigmoid-like mapping to determine exploration weight
    # When f_best is large, we want more exploration (higher w_ucb)
    # When f_best is small, we want more exploitation (lower w_ucb)
    
    # Normalize f_best: assume typical range
    # For Ackley/Rastrigin/Levy: optimum ~0, initial values can be large
    # For Hartmann: optimum ~-3.86
    # Use a general heuristic: if f_best > 1, we're early; if f_best < 0.5, we're late
    # Smooth transition around f_best = 0.5
    
    # Compute w_ucb: weight for the UCB component
    # w_ucb = 1.0 means pure cost-aware UCB (more exploratory)
    # w_ucb = 0.0 means pure cost-aware EI (more exploitative)
    
    # Use a logistic function centered at f_best = 0.5
    # When f_best is large (e.g., 10), w_ucb ~ 1.0 (explore)
    # When f_best is small (e.g., 0.1), w_ucb ~ 0.3 (exploit)
    
    # Clip f_best to a reasonable range for the logistic
    f_best_clipped = np.clip(f_best, -5.0, 20.0)
    
    # Logistic: w_ucb = 1 / (1 + exp(k * (f_best - c)))
    # Choose k = 2.0, c = 1.0
    # At f_best = 1: w_ucb = 0.5
    # At f_best = 3: w_ucb ~ 0.047 (exploit)
    # At f_best = 5: w_ucb ~ 0.0003 (exploit)
    # At f_best = -1: w_ucb ~ 0.95 (explore)
    # At f_best = -3: w_ucb ~ 0.999 (explore)
    
    k = 2.0
    c = 1.0
    w_ucb = 1.0 / (1.0 + np.exp(k * (f_best_clipped - c)))
    
    # Clamp to [0.2, 0.8] to avoid extremes
    w_ucb = np.clip(w_ucb, 0.2, 0.8)
    
    # Compute the hybrid score
    score = _hybrid_score(mu, sigma, f_best, cost, w_ucb)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
