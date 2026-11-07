"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Pre-computed constants for speed
_N = 2000
_DEADLINE = time.time() + 325.0

def _norm_cdf(x):
    """Fast normal CDF using scipy."""
    return norm.cdf(x)

def _norm_pdf(x):
    """Fast normal PDF using scipy."""
    return norm.pdf(x)


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next).
    
    mu, sigma: arrays of shape (2000,) in objective units
    f_best: scalar, lowest observed value
    cost: array of shape (2000,), evaluation cost per candidate
    
    Uses a structural change: Thompson Sampling with cost-aware reweighting.
    Instead of deterministic EI/UCB, we sample from the posterior and use
    a hard feasibility gate based on cost quantiles to concentrate search
    in low-cost regions until GP uncertainty there is exhausted.
    """
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        return np.ones(_N)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Cost-aware Thompson Sampling:
    # Sample from N(mu, sigma^2) and score by -sample / cost
    # This naturally balances exploration (high sigma -> wider sample range)
    # and exploitation (low mu -> lower samples)
    
    # Use a fixed random seed for reproducibility within the call
    # Actually, we should use a different seed each call for proper TS
    # But since we're called repeatedly, let's use a global counter
    global _ts_counter
    try:
        _ts_counter
    except NameError:
        _ts_counter = 0
    _ts_counter += 1
    
    rng = np.random.RandomState(_ts_counter % (2**31 - 1))
    sample = rng.normal(mu, sigma_safe)
    
    # Score: lower sample is better, divide by cost for cost-awareness
    cost_safe = np.maximum(cost, 1e-8)
    ts_score = -sample / cost_safe
    
    # Hard feasibility gate: only consider candidates in the bottom 50% of cost
    # This forces concentration in low-cost regions
    cost_q50 = np.percentile(cost, 50)
    gate_mask = cost <= cost_q50
    
    # Apply strong penalty to high-cost candidates
    # Use a multiplicative gate: high cost candidates get score * 0.1
    score = np.where(gate_mask, ts_score, ts_score * 0.1)
    
    # Additionally, apply a soft bonus for candidates in the bottom 25% of cost
    cost_q25 = np.percentile(cost, 25)
    low_cost_mask = cost <= cost_q25
    score = np.where(low_cost_mask, score * 1.5, score)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Add a small deterministic EI component to stabilize early iterations
    # when TS samples are noisy
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * _norm_cdf(z) + sigma_safe * _norm_pdf(z)
    ei = np.maximum(ei, 0.0)
    ei_cost = ei / cost_safe
    
    # Normalize EI to comparable scale
    ei_max = np.max(ei_cost) if len(ei_cost) > 0 else 1.0
    ei_norm = ei_cost / (ei_max + 1e-12)
    
    # Normalize TS score
    ts_max = np.max(score) if len(score) > 0 else 1.0
    ts_min = np.min(score) if len(score) > 0 else 0.0
    ts_range = ts_max - ts_min if ts_max > ts_min else 1.0
    ts_norm = (score - ts_min) / (ts_range + 1e-12)
    
    # Blend: 70% TS, 30% EI for stability
    final_score = 0.7 * ts_norm + 0.3 * ei_norm
    
    # Re-apply the hard cost gate to final score
    final_score = np.where(gate_mask, final_score, final_score * 0.2)
    
    # Ensure finite and non-negative
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    final_score = np.maximum(final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
