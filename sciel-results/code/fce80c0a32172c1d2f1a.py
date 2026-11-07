"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard: if we're running long, return a simple score
    elapsed = time.time() - _start_time
    if elapsed > 290:
        # Fallback: simple UCB with conservative exploration
        return -mu + 1.0 * np.maximum(sigma, 1e-9)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Compute standard acquisition functions
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # UCB with adaptive beta that decreases over time
    # Early: more exploration, Later: more exploitation
    total_expected_calls = 30  # 5 initial + 25 more
    progress = min(_call_count / total_expected_calls, 1.0)
    # beta starts at 2.5 and decays to 0.5
    beta = 2.5 - 2.0 * progress
    ucb = -mu + beta * sigma_safe
    
    # Thompson Sampling: sample from posterior and pick the argmin
    # This is a different structural approach - randomized
    # Use multiple samples to get a robust estimate
    n_samples = 5
    theta_samples = np.random.normal(mu, sigma_safe, size=(n_samples, len(mu)))
    # For each candidate, compute the probability it would be the minimum across samples
    # A candidate is "good" if it frequently comes up as the minimum
    is_min = np.argmin(theta_samples, axis=0) == np.arange(len(mu))[None, :]
    # Actually, let's do this properly
    mins = np.min(theta_samples, axis=0)  # This doesn't work directly
    
    # Better Thompson sampling approach:
    # For each candidate, the probability that it's the best among all candidates
    # given a random draw from its posterior
    # This is expensive, so use a simpler variant:
    # Score = -E[min over a few random draws]
    # Actually, let's use a simpler but effective approach:
    # Weighted combination with Thompson-sampling-inspired exploration
    
    # Let's use a different structural approach: Knowledge Gradient
    # KG approximates the expected value of information from evaluating each point
    # Simplified KG: how much does evaluating this point improve our knowledge?
    # For 1D BO, a good proxy is the reduction in expected minimum
    
    # For practical purposes, let's use a blend that adapts based on confidence
    # When sigma is high relative to f_best range, explore more
    
    # Scale EI to be comparable to UCB
    ei_range = np.max(ei) - np.min(ei) if np.max(ei) > np.min(ei) else 1.0
    ucb_range = np.max(ucb) - np.min(ucb) if np.max(ucb) > np.min(ucb) else 1.0
    
    # Normalize both to [0, 1]
    ei_norm = (ei - np.min(ei)) / (ei_range + 1e-12)
    ucb_norm = (ucb - np.min(ucb)) / (ucb_range + 1e-12)
    
    # Blend: weight shifts from exploration to exploitation over time
    # Early: 50% EI, 50% UCB (exploration-heavy)
    # Late: 80% EI, 20% UCB (exploitation-heavy)
    w_ei = 0.5 + 0.3 * progress
    w_ucb = 1.0 - w_ei
    
    score = w_ei * ei_norm * ei_range + w_ucb * ucb_norm * ucb_range
    
    # Add a small random component for diversity (Thompson-sampling-like)
    # This helps escape local optima in the search space
    noise_scale = 0.01 * (1.0 - progress)
    score = score + noise_scale * np.random.randn(len(mu)) * np.std(score)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
