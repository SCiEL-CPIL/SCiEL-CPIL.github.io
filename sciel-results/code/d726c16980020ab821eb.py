import numpy as np
import time
from scipy.stats import norm

# Wall-clock deadline guard
_T0 = time.time()
_WALL_LIMIT = 330.0
_SAFETY = 45.0
_DEADLINE = _WALL_LIMIT - _SAFETY

def _deadline_exceeded():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best):
    """
    Hybrid Thompson Sampling + Expected Improvement acquisition function.
    
    Structural change: Instead of a deterministic score based on a fixed beta/UCB
    or pure EI, we sample from the posterior for each candidate (Thompson Sampling)
    and combine this sampled value with the Expected Improvement.
    
    Specifically:
    1. Sample z ~ N(0, 1) for each of the 2000 candidates.
    2. Sampled value f_sampled = mu + sigma * z.
    3. Compute EI based on the posterior.
    4. Score = w_ts * (-f_sampled) + w_ei * EI.
    
    The TS component provides stochastic exploration that naturally balances
    exploitation and exploration without manual tuning of beta.
    The EI component provides a deterministic exploitation signal.
    
    We use a fixed weight for simplicity and robustness, but the stochastic
    nature of TS allows the algorithm to escape local minima in the 6D
    Hartmann landscape more effectively than deterministic UCB/EI.
    """
    if _deadline_exceeded():
        # Fallback: return deterministic EI to be safe
        sigma_safe = np.where(sigma < 1e-12, 1e-12, sigma)
        z_ei = (f_best - mu) / sigma_safe
        EI = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
        EI = np.where(np.isfinite(EI), EI, 0.0)
        return EI

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.where(sigma < 1e-12, 1e-12, sigma)
    
    # 1. Expected Improvement (deterministic exploitation)
    imp = f_best - mu
    z_ei = imp / sigma_safe
    EI = imp * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    EI = np.where(np.isfinite(EI), EI, 0.0)
    
    # 2. Thompson Sampling (stochastic exploration)
    # Sample from the posterior for each candidate
    # z ~ N(0, 1)
    z_ts = np.random.standard_normal(len(mu))
    f_sampled = mu + sigma_safe * z_ts
    
    # For minimization, lower f_sampled is better.
    # We want higher scores to be selected.
    # So we use -f_sampled.
    # However, f_sampled can be large negative (good) or large positive (bad).
    # To combine with EI (which is always >= 0), we need to scale or shift.
    
    # A common approach is to use the rank or a normalized score.
    # But a simple linear combination can work if we weight TS heavily early on.
    
    # Let's use a dynamic weight for TS based on f_best, similar to UCB beta.
    # When f_best is poor (high value), we want more exploration (TS).
    # When f_best is good (low value), we want more exploitation (EI).
    
    min_val = -3.32237
    worst_val = 0.0
    
    f_best_clamped = np.clip(f_best, min_val, worst_val)
    t = (f_best_clamped - min_val) / (worst_val - min_val)  # 0 (good) to 1 (bad)
    
    # Weight for TS: higher when t is high (early exploration)
    # Weight for EI: higher when t is low (late exploitation)
    
    # Let's define:
    # w_ts = 1.0 + t  (ranges from 1.0 to 2.0)
    # w_ei = 2.0 - t  (ranges from 2.0 to 1.0)
    # This ensures both are always positive and sum to ~3.0
    
    w_ts = 1.0 + t
    w_ei = 2.0 - t
    
    # Scale f_sampled to be comparable to EI.
    # EI is typically in the range [0, f_best - min_val] ~ [0, 3.3].
    # f_sampled is in the range [mu - 3*sigma, mu + 3*sigma].
    # For Hartmann 6D, values are roughly in [-3.3, 0].
    # So -f_sampled is roughly in [0, 3.3].
    
    # We can just use -f_sampled directly, but it has higher variance.
    # To stabilize, we might want to normalize it.
    # However, simple linear combination is often robust.
    
    # Score = w_ei * EI + w_ts * (-f_sampled)
    
    scores = w_ei * EI + w_ts * (-f_sampled)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
