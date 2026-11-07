"""Thompson Sampling acquisition for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 290.0  # 10s safety margin before 300s kill

def acquisition(mu, sigma, f_best):
    global _start_time, _DEADLINE

    # Deadline guard: if we're running out of time, return a neutral score
    if time.time() > _DEADLINE:
        return np.ones_like(mu) * 0.0

    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-10)

    # Thompson Sampling: draw one sample from the posterior at each candidate
    # f(x) ~ N(mu, sigma^2)
    # We minimize, so we want the lowest sampled value
    # Score = -sample (higher score = lower sampled value = better)
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(mu.shape)

    # Return negative samples as scores (higher score = lower f value = better)
    scores = -samples

    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    return scores
# EVOLVE-BLOCK-END
