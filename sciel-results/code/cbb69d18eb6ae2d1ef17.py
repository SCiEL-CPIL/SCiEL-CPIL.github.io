"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Hybrid PoI + Thompson Sampling acquisition with cost awareness.
    
    Combines the robustness of Probability of Improvement (deterministic,
    scores by P(f(x) < f_best)) with stochastic exploration from Thompson
    Sampling. The TS sample perturbs the improvement probability, providing
    natural exploration without hand-tuned constants. A small uncertainty
    bonus encourages coverage of unexplored regions.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: scalar - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
        scores: array of shape (2000,) - higher means evaluate next
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        return 1.0 / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Base score: Probability of Improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    poi = np.clip(poi, 0.0, 1.0)
    poi = np.where(np.isfinite(poi), poi, 0.0)
    
    # Thompson Sampling perturbation: sample a single function realization
    # and compute the improvement from that sample
    # Use a deterministic seed based on f_best for reproducibility per call
    seed = int(np.abs(f_best * 1000000)) % (2**32 - 1)
    rng = np.random.RandomState(seed)
    
    # Sample function values: f(x) ~ N(mu, sigma^2)
    f_sample = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # Improvement from the sampled function: max(f_best - f_sample, 0)
    # Normalize by a scale to make it comparable to PoI (which is in [0,1])
    # Use the typical scale of improvements: f_best itself or a fixed scale
    # Griewank values are small (0 to ~0.5), so use a reasonable normalization
    improvement_scale = max(abs(f_best), 0.1)  # avoid division by zero
    ts_improvement = np.maximum(f_best - f_sample, 0.0) / improvement_scale
    ts_improvement = np.clip(ts_improvement, 0.0, 1.0)
    ts_improvement = np.where(np.isfinite(ts_improvement), ts_improvement, 0.0)
    
    # Blend PoI and TS improvement: 70% PoI (robust) + 30% TS (exploration)
    # This gives a stable baseline with stochastic exploration
    blended = 0.7 * poi + 0.3 * ts_improvement
    
    # Small uncertainty bonus to encourage exploration of unexplored regions
    # Normalize sigma to [0, 1] using a reasonable scale
    sigma_norm = sigma_safe / (sigma_safe.max() + eps) if sigma_safe.max() > 0 else np.zeros_like(sigma_safe)
    uncertainty_bonus = 0.1 * sigma_norm
    
    # Final score before cost normalization
    raw_score = blended + uncertainty_bonus
    raw_score = np.where(np.isfinite(raw_score), raw_score, 0.0)
    
    # Cost-aware: divide by cost (higher cost = lower score)
    # cost = 1 + mean(x_scaled), ranges from 1 to 2
    scores = raw_score / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
