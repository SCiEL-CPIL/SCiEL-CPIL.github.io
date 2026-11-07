"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_T0 = None
_DEADLINE = 325.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling for Bayesian optimization
    of rastrigin 2d. Thompson Sampling provides stochastic exploration that escapes
    local minima more effectively than deterministic EI/UCB in multimodal landscapes.
    """
    global _T0
    if _T0 is None:
        _T0 = time.time()

    # Deadline guard
    if time.time() - _T0 > _DEADLINE:
        # Simple fallback: prefer low mu, high sigma, low cost
        safe_mu = np.where(np.isfinite(mu), mu, 0.0)
        safe_sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
        safe_cost = np.where(np.isfinite(cost), cost, 1.0)
        safe_sigma = np.maximum(safe_sigma, 1e-8)
        safe_cost = np.maximum(safe_cost, 1e-8)
        fallback_score = (-safe_mu + 2.0 * safe_sigma) / safe_cost
        return np.where(np.isfinite(fallback_score), fallback_score, 0.0)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)

    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)

    # Thompson Sampling: sample from the posterior for each candidate
    # For minimization, we sample from N(mu, sigma) and pick the lowest
    # But since we need a deterministic score per candidate, we use the
    # expected value of the minimum of K samples (a proxy for TS)
    
    # Adaptive number of samples based on uncertainty
    mean_sigma = np.mean(sigma)
    
    # Use a fixed number of samples for stability, but weight by uncertainty
    # Sample once per candidate: the TS score is the sampled value
    # We want to maximize the probability of finding a new minimum,
    # which correlates with low sampled values
    
    # Draw one sample per candidate from N(mu, sigma)
    # Use a fixed seed for reproducibility within a call, but vary across calls
    # Actually, we should use random sampling for true TS behavior
    rng = np.random.default_rng()
    samples = rng.normal(mu, sigma)
    
    # The TS score: we want to evaluate the point with the lowest sampled value
    # So score = -sample (higher is better)
    # But we also need cost-awareness
    
    # Cost-awareness: divide by cost to penalize expensive evaluations
    ts_score = -samples / cost
    
    # Add a small exploitation component based on EI to ensure we don't
    # ignore known good regions entirely
    improvement = f_best - mu
    z = improvement / sigma
    z = np.clip(z, -10, 10)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI
    ei_max = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    ei_normalized = ei / (ei_max + 1e-10)
    
    # Combine: primarily Thompson Sampling, with a small EI component
    # Weight shifts toward EI as f_best approaches 0
    if f_best < 0.5:
        ts_weight = 0.7
        ei_weight = 0.3
    elif f_best < 2.0:
        ts_weight = 0.8
        ei_weight = 0.2
    else:
        ts_weight = 0.9
        ei_weight = 0.1
    
    score = ts_weight * ts_score + ei_weight * ei_normalized / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
