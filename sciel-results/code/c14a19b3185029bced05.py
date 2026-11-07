"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = _start_time + 280  # 20s safety margin before 300s kill


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return best-guess: lowest mu / cost as a fallback
        return (f_best - mu) / (cost + 1e-9)

    # Thompson Sampling-based acquisition
    # Sample from the GP predictive: y = mu + sigma * z, where z ~ N(0,1)
    # We want to minimize f, so we want low sampled values.
    # Score = -sampled_value / cost (higher is better)
    # But we need to be careful: we want to explore where uncertainty is high
    # and exploit where the function is likely low.
    
    # Use a single random draw per candidate from the predictive
    # To make it cost-aware, we normalize by cost
    # We want high scores for candidates that are likely to have low f and low cost
    
    # Sample: y_k = mu_k + sigma_k * z_k
    # Score: we want to maximize (f_best - y_k) / cost_k, but y_k is random
    # Actually, for Thompson Sampling in minimization:
    # We sample y_k and pick the one with lowest y_k / cost_k
    # But we return scores where higher = evaluate next
    # So score_k = -y_k / cost_k (negate because we want to minimize y_k)
    # Or equivalently: score_k = (f_best - y_k) / cost_k doesn't quite work
    
    # Let me think again. In TS for minimization:
    # 1. Sample y_k ~ N(mu_k, sigma_k^2) for each candidate
    # 2. Pick candidate with minimum y_k / cost_k
    # 
    # Since the evaluator picks the candidate with highest score, we want:
    # score_k should be high when y_k / cost_k is low
    # So: score_k = -y_k / cost_k
    # 
    # But we need to ensure finite values and handle edge cases
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw standard normal samples
    z = np.random.standard_normal(len(mu))
    
    # Sampled function values
    y = mu + sigma_safe * z
    
    # Cost-aware score: we want to minimize y / cost
    # So score = -y / cost (higher is better)
    scores = -y / (cost + 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
