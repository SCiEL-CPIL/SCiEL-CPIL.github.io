# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = 325.0  # 5s safety margin before 330s
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _matern52_kernel(X1, X2, length_scale=0.2):
    """Matern-5/2 kernel for standardized coordinates."""
    # Compute pairwise distances
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Matern-5/2: (1 + sqrt(3)*d/l + 3*d^2/l^2) * exp(-sqrt(3)*d/l)
    r = np.sqrt(3.0) * d / length_scale
    return (1.0 + r + r * r / 3.0) * np.exp(-r)

def _compute_acquisition_scores(mu, sigma, f_best, cost, deadline_margin=10.0):
    """
    Compute a combined acquisition score using:
    1. Cost-normalized Expected Improvement (EI)
    2. Thompson Sampling-like exploration via uncertainty weighting
    3. Adaptive weighting based on cost budget remaining
    
    Returns scores where higher = better candidate to evaluate next.
    """
    n = len(mu)
    
    # Guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost normalization: prefer cheaper points
    cost_safe = np.maximum(cost, 1e-6)
    ei_cost = ei / cost_safe
    
    # Thompson Sampling-inspired exploration: sample from posterior
    # Use a deterministic proxy: weight by sigma (uncertainty) with a
    # temperature that decreases as we spend more budget
    total_cost = np.sum(cost_safe)
    # As budget is spent, shift from exploration to exploitation
    # Early: more exploration weight; Late: more exploitation weight
    exploration_weight = np.clip(1.0 - total_cost / 40.0, 0.1, 1.0)
    
    # Exploration score: prefer high uncertainty, low cost
    exploration = sigma_safe / cost_safe
    
    # Combine: weighted blend of EI/cost and exploration/cost
    # Use a softmax-like normalization to keep scores comparable
    scores = (1.0 - exploration_weight) * ei_cost + exploration_weight * exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Add a small bonus for points with very low cost to encourage
    # exploring cheap regions early
    cost_bonus = 0.1 / cost_safe
    scores += cost_bonus * exploration_weight
    
    return scores

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean predictions from GP
    sigma : array of shape (2000,)
        Standard deviation predictions from GP
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline check
    if _elapsed() > _DEADLINE:
        # Return a safe fallback: prefer low cost
        cost_safe = np.maximum(cost, 1e-6)
        return 1.0 / cost_safe
    
    # Validate inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Compute scores
    scores = _compute_acquisition_scores(mu, sigma, f_best, cost)
    
    # Ensure output is finite and has correct shape
    scores = np.where(np.isfinite(scores), scores, 0.0)
    if scores.shape != (n,):
        scores = scores.reshape(n)
    
    return scores
# EVOLVE-BLOCK-END
