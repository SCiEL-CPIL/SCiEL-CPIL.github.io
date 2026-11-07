"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_START_TIME = time.time()
_TIME_LIMIT = 295.0  # Safety margin below 330s wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Explicit deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Fallback: simple cost-aware EI
        m = np.asarray(mu, dtype=np.float64)
        s = np.asarray(sigma, dtype=np.float64)
        c = np.asarray(cost, dtype=np.float64)
        s = np.maximum(s, 1e-9)
        c = np.maximum(c, 1e-9)
        z = (f_best - m) / s
        z = np.clip(z, -10, 10)
        ei = (f_best - m) * norm.cdf(z) + s * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.where(np.isfinite(ei / c), ei / c, 0.0)
    
    # Convert to numpy arrays for efficiency
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma and cost are positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Thompson Sampling Component ---
    # Draw a single sample from the posterior at each candidate point.
    # Use a fixed seed based on the current f_best to ensure reproducibility
    # across calls with the same state, but vary it slightly if f_best changes.
    # Actually, for Thompson sampling in a loop, we want a fresh random draw
    # each time the function is called, unless we are in a deterministic mode.
    # However, the task says "seeded instances" for the function f_k, not for the acquisition.
    # The acquisition should be stochastic. Let's use a global RNG or just np.random.
    # To be safe and reproducible for debugging, we can seed based on a hash of inputs?
    # No, standard TS uses a fresh sample.
    
    # We will use np.random.default_rng() but we need to be careful about state.
    # Since we don't have access to external state, we just use the global np.random.
    # To make it slightly more stable, we can use a deterministic seed if we want,
    # but TS is inherently stochastic.
    
    # Sample from N(mu, sigma^2)
    # Note: np.random.normal is fast.
    ts_samples = np.random.normal(mu, sigma_safe)
    
    # Score for Thompson Sampling: we want to minimize the sampled value.
    # So higher score is better -> -ts_samples / cost
    # This balances exploration (high sigma leads to wide spread, chance of low sample)
    # and exploitation (low mu leads to low samples).
    ts_score = -ts_samples / cost_safe
    
    # --- LCB Component (from Inspiration 1, scored 1.08) ---
    # LCB provides a deterministic exploration bonus.
    # Adaptive beta: scale with max uncertainty.
    max_sigma = np.max(sigma_safe)
    beta = 2.0 * max_sigma
    
    # LCB score (for minimization, we maximize -mu + beta*sigma)
    lcb_score = -mu + beta * sigma_safe
    
    # Cost-aware LCB
    lcb_cost = lcb_score / cost_safe
    
    # --- Blending ---
    # The current program (0.68) used 0.5 EI + 0.5 LCB.
    # Inspiration 1 (1.08) used pure LCB with cost penalty: -mu + beta*sigma - gamma*cost.
    # Note: -mu + beta*sigma - gamma*cost is NOT the same as (-mu + beta*sigma) / cost.
    # Let's analyze Inspiration 1's formula: score = -mu + beta*sigma - gamma*cost.
    # This subtracts a linear cost penalty.
    # My current lcb_cost is (-mu + beta*sigma) / cost.
    
    # Let's try to combine the stochastic nature of TS with the robustness of LCB.
    # TS can be noisy. LCB is stable.
    # A weighted average of the two scores (both normalized by cost) seems reasonable.
    
    # Let's refine the LCB part to match Inspiration 1's success more closely.
    # Inspiration 1: score = -mu + beta*sigma - gamma*cost
    # This is equivalent to maximizing (-mu + beta*sigma - gamma*cost).
    # If we want to blend with TS, we should normalize them to similar scales.
    
    # Let's use a blend of:
    # 1. TS score: -sample / cost
    # 2. LCB score: (-mu + beta*sigma) / cost
    
    # Weights: Start with 0.5/0.5, but maybe weight LCB more since it scored better.
    # Let's try 0.4 * TS + 0.6 * LCB.
    
    scores = 0.4 * ts_score + 0.6 * lcb_cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
