import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 5.0
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    _eval_count += 1
    
    # Deadline guard: return safe scores if approaching time limit
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # If we are at the deadline, return a neutral score to force a selection quickly
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: sample from the posterior
    # Use a fixed seed per call to ensure reproducibility if needed, but random sampling is standard
    # We sample once per candidate
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # The core Thompson Sampling score is simply the negative of the sample
    # Since we want to minimize the function, the best candidate has the lowest sample value.
    # However, the acquisition function must return scores where HIGHER is better.
    # So we return -sample.
    ts_score = -samples
    
    # To improve upon pure TS which can be too exploratory or get stuck,
    # we blend with an exploitation term (Expected Improvement) that focuses on the best known region.
    # This creates a hybrid: "Thompson Sampling with Exploitation Bias".
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to [0, 1] for blending
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-10
    ei_normalized = (ei - ei_min) / ei_range
    
    # Normalize TS score to [0, 1] for blending
    ts_min = np.min(ts_score)
    ts_max = np.max(ts_score)
    ts_range = ts_max - ts_min + 1e-10
    ts_normalized = (ts_score - ts_min) / ts_range
    
    # Adaptive blending:
    # Early on (low _eval_count), rely more on Thompson Sampling for broad exploration.
    # Later on (high _eval_count), rely more on EI for precise exploitation.
    # The current program used a sigmoid based on mean_sigma, which is a good proxy for "how much do we know?".
    # But _eval_count is a direct measure of progress.
    # Let's use a decay: weight_ei increases as we progress.
    
    # Start with 80% TS, 20% EI. Decay to 20% TS, 80% EI over ~20 evaluations.
    # This ensures we explore early and exploit late.
    ts_weight = 0.2 + 0.8 * np.exp(-_eval_count / 15.0)
    ei_weight = 1.0 - ts_weight
    
    # Blend the normalized scores
    score = ts_weight * ts_normalized + ei_weight * ei_normalized
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
