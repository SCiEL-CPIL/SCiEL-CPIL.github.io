"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Per-process deadline guard
_T0 = time.time()
_DEADLINE = 330.0

def _elapsed():
    return time.time() - _T0

def acquisition(mu, sigma, f_best, cost):
    # Guard: if we are dangerously close to the wall-clock limit, return a
    # deterministic, finite, low-risk signal immediately.
    if _elapsed() > _DEADLINE - 1.0:
        return np.zeros(np.shape(mu), dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Sanitize inputs
    # sigma should be positive. GP noise is 1e-6, so sigma shouldn't be 0, but just in case.
    sigma = np.clip(sigma, 1e-6, None)
    cost = np.clip(cost, 1e-6, None)
    f_best = float(f_best) if np.isfinite(f_best) else 0.0

    # Thompson Sampling: sample one objective value from the Gaussian posterior for each candidate
    # sample = mu + sigma * epsilon
    # We use a fixed seed for reproducibility? No, the function is called multiple times.
    # Using a global RNG or local RNG.
    # To be fast, we can generate random numbers.
    # However, TS with a single sample can be noisy.
    # But the prompt suggests: "Model the acquisition as a Thompson Sampling score by sampling a single objective value from the Gaussian posterior for each candidate, normalized by the evaluation cost"
    
    # Let's use a global RNG to avoid seeding issues, or just use np.random.
    # To ensure we don't have side effects that might be problematic if the function is called in parallel (it's not, it's a loop),
    # but usually in BO, acquisition is called sequentially.
    # We will use a module-level RNG.
    global _rng
    if not hasattr(acquisition, '_rng'):
        acquisition._rng = np.random.default_rng()
    
    eps = acquisition._rng.standard_normal(mu.shape)
    sample = mu + sigma * eps
    
    # For minimization, we want low values.
    # Score should be high for good candidates.
    # So we want to minimize 'sample'.
    # Score = -sample / cost
    
    # To avoid extreme negative scores (which are good), we might need to be careful.
    # But higher score = evaluate next.
    # So if sample is very low (good), -sample is very high.
    # If sample is very high (bad), -sample is very low (negative).
    
    score = -sample / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
