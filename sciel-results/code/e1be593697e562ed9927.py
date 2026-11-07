import numpy as np
import time
from scipy.special import erf

# Global deadline guard: 330s wall-clock limit, use 320s as safety margin
_start_time = time.time()
_DEADLINE = _start_time + 320.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Combines Thompson Sampling (TS) for natural exploration-exploitation balance
    with a Knowledge Gradient (KG) inspired term that explicitly models the
    expected reduction in the minimum observed value. TS provides stochastic
    diversity to escape local minima, while the KG term provides a deterministic
    signal pointing toward regions with high probability of containing the
    global minimum.
    
    Args:
        mu: array of shape (2000,) - posterior mean at candidates
        sigma: array of shape (2000,) - posterior std at candidates
        f_best: float - best observed function value so far
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Deadline guard: if we're too close to the limit, use a fast fallback
    if time.time() > _DEADLINE - 5.0:
        # Fast fallback: simple LCB for minimization
        return -(mu - 0.5 * sigma_safe)
    
    # --- Thompson Sampling Component ---
    # Create a deterministic seed from the optimization state for reproducibility
    # within a single acquisition call, but varying across calls as f_best and mu change
    seed_raw = abs(f_best) * 1e6 + np.sum(np.abs(mu)) + np.sum(sigma)
    seed = int(seed_raw) % (2**32)
    
    rng = np.random.default_rng(seed)
    
    # Draw standard normal samples
    eps = rng.standard_normal(mu.shape[0])
    
    # Thompson Sampling: Draw from the posterior distribution N(mu, sigma^2)
    samples = mu + sigma_safe * eps
    
    # For minimization, lower samples are better.
    # Score = -sample (higher score means we want to evaluate this point)
    ts_scores = -samples
    
    # --- Knowledge Gradient (KG) Inspired Component ---
    # The Knowledge Gradient measures the expected reduction in the minimum observed value.
    # For a candidate x with posterior N(mu_x, sigma_x^2), the expected improvement
    # in the global minimum is related to the probability that the true value at x
    # is lower than f_best, weighted by the magnitude of that improvement.
    #
    # A simplified analytical approximation for the KG in this context:
    # KG(x) = E[max(f_best - Y_x, 0)] where Y_x ~ N(mu_x, sigma_x^2)
    # This is exactly the Expected Improvement (EI) formula.
    # However, to distinguish from pure EI and add the "global optimality" flavor,
    # we can weight this by the uncertainty itself, favoring regions where the
    # posterior is wide enough that a significant improvement is possible.
    #
    # Let's compute EI first:
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -50, 50)
    
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    phi_z = np.exp(-0.5 * z_clipped ** 2) / sqrt_2pi
    Phi_z = 0.5 * (1.0 + erf(z_clipped / np.sqrt(2.0)))
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # To make this more "KG-like" and less prone to over-exploitation,
    # we can use a version that accounts for the fact that we have a limited budget.
    # A common heuristic is to scale EI by the inverse of the number of remaining
    # evaluations, but since we don't have that directly, we use sigma as a proxy
    # for "potential".
    #
    # Let's define the KG-inspired score as EI * (1 + sigma_norm)
    # This boosts EI in regions of higher uncertainty, similar to how KG values
    # information gain.
    
    # Normalize EI to zero mean, unit variance for fair combination with TS
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std < 1e-9:
        z_ei = np.zeros_like(ei)
    else:
        z_ei = (ei - ei_mean) / ei_std
    
    # Normalize TS scores to zero mean, unit variance
    ts_mean = np.mean(ts_scores)
    ts_std = np.std(ts_scores)
    if ts_std < 1e-9:
        z_ts = np.zeros_like(ts_scores)
    else:
        z_ts = (ts_scores - ts_mean) / ts_std
    
    # Combine TS and KG-inspired EI
    # We want a balance. TS is stochastic, KG is deterministic.
    # Early on, TS helps explore. Later, KG helps exploit.
    # We use mean sigma as a proxy for exploration level.
    
    mean_sigma = np.mean(sigma_safe)
    sigma_norm = np.clip(mean_sigma / 0.3, 0.0, 1.0)
    
    # Weights: early = more TS (exploration), late = more KG (exploitation)
    w_ts = 0.4 + 0.3 * sigma_norm  # ranges from 0.4 to 0.7
    w_kg = 1.0 - w_ts
    
    scores = w_ts * z_ts + w_kg * z_ei
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
