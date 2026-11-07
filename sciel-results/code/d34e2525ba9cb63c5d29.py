"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds of safety margin


def _elapsed():
    return time.time() - _start_time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Strategy: Monte Carlo Expected Improvement (MC-EI) + Exploitation.
    
    Instead of using the analytic EI formula which can suffer from numerical
    instability for large |z| or require careful handling of the normal CDF/PDF,
    we estimate the Expected Improvement via Monte Carlo sampling.
    
    We sample K random values from the predicted Gaussian distribution N(mu, sigma^2)
    and compute the average improvement over f_best. This is robust and simple.
    
    We then combine this MC-EI score with a strong exploitation term (-mu) to
    ensure that regions with very low predicted means are prioritized, even if
    their uncertainty is low. This creates a "best-of-two" dynamic:
    1. If a region has a very low mean, -mu will be high.
    2. If a region has high potential for improvement (high EI), MC-EI will be high.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    # Deadline check
    if _elapsed() > _DEADLINE_MARGIN:
        # Return a safe fallback: just negative mu (exploitation only)
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        scores = -mu
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # --- Monte Carlo Expected Improvement ---
    # We want to estimate E[max(f_best - f, 0)] where f ~ N(mu, sigma^2)
    # This is equivalent to E[max(f_best - f, 0)]
    
    # Number of samples per point. 
    # 100 samples is a good trade-off between accuracy and speed for 2000 points.
    # Total operations: 2000 * 100 = 200,000, which is very fast in NumPy.
    K = 100
    
    # Generate random samples: shape (K, n)
    # Standard normal samples
    z = np.random.standard_normal((K, n))
    
    # Transform to predicted values: f_sample = mu + sigma * z
    # mu and sigma are (n,), so we broadcast
    f_samples = mu[None, :] + sigma_safe[None, :] * z
    
    # Compute improvement: max(f_best - f_samples, 0)
    improvements = np.maximum(f_best - f_samples, 0.0)
    
    # Average over samples to get MC-EI estimate: shape (n,)
    mc_ei = np.mean(improvements, axis=0)
    
    # --- Exploitation Term ---
    # We want to reward low mu values.
    # Simple approach: -mu
    # However, to balance scales, we might normalize or weight.
    # Since mu and mc_ei are in objective units, we can combine them directly
    # with a weight.
    
    # Let's use a weighted sum:
    # score = w_exploit * (-mu) + w_ei * mc_ei
    
    # Adaptive weights?
    # Early on, sigma is large, so mc_ei is large. -mu might be noisy.
    # Late on, sigma is small, mc_ei is small, -mu dominates.
    # This seems naturally adaptive.
    
    # Let's try a fixed weight first to see the behavior.
    # If we use equal weights, the scales might be different.
    # Typical Goldstein-Price values are ~3 to 40.
    # f_best starts around 10-20 and drops to 3.
    # mu will be in similar range.
    # -mu will be negative. Wait, higher score is better.
    # If mu is low (good), -mu is high (positive).
    # If mu is high (bad), -mu is low (negative).
    # mc_ei is always positive.
    
    # To ensure the acquisition is positive and comparable, let's shift -mu.
    # Or just use -mu directly and add mc_ei.
    
    # Let's normalize both to [0, 1] range for stable combination?
    # Or just sum them?
    
    # If we sum them:
    # Point A: mu=3, sigma=0.1 -> -mu = -3, mc_ei ~ 0. Score ~ -3.
    # Point B: mu=10, sigma=2.0 -> -mu = -10, mc_ei ~ 1. Score ~ -9.
    # Point A is preferred. This is correct exploitation.
    
    # Point C: mu=15, sigma=5.0 -> -mu = -15, mc_ei ~ 2. Score ~ -13.
    # Point B is preferred.
    
    # What if Point D: mu=12, sigma=0.01 -> -mu = -12, mc_ei ~ 0. Score ~ -12.
    # Point B (mu=10, sigma=2) has score -9. Point B is preferred.
    # But Point D is very certain it's 12. Point B is uncertain but mean is 10.
    # If f_best is 10, Point D has no improvement chance. Point B has some.
    # So B > D is correct.
    
    # What if f_best is 11?
    # Point D: mu=12, sigma=0.01. -mu = -12. mc_ei = max(11-12, 0) = 0. Score -12.
    # Point B: mu=10, sigma=2. -mu = -10. mc_ei > 0. Score > -10.
    # B is preferred. Correct.
    
    # So simple sum seems reasonable.
    
    scores = -mu + mc_ei
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Final deadline check
    if _elapsed() > _DEADLINE_MARGIN:
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
