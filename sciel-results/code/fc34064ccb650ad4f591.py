# EVOLVE-BLOCK-START
import time
import numpy as np
from math import erf, sqrt, pi

_T0 = time.time()
_DEADLINE = _T0 + 320.0  # safety margin vs 330s kill

_INV_SQRT2PI = 1.0 / sqrt(2.0 * pi)
_INV_SQRT2 = 1.0 / sqrt(2.0)

def _norm_cdf_vec(z):
    # Vectorized normal CDF using erf
    return 0.5 * (1.0 + np.vectorize(erf, otypes=[float])(z * _INV_SQRT2))

def _norm_pdf_vec(z):
    return _INV_SQRT2PI * np.exp(-0.5 * z * z)

def _norm_ppf(p):
    """
    Approximate inverse CDF (PPF) for standard normal.
    We need z_alpha such that P(Z < z_alpha) = alpha.
    Since alpha is small (e.g., 1e-4 to 1e-2), z_alpha is negative.
    We can use the symmetry: P(Z < -z) = P(Z > z) = 1 - Phi(z).
    So if we want Phi(z_alpha) = alpha, let z_alpha = -z_p where Phi(z_p) = 1-alpha.
    For small alpha, 1-alpha is close to 1.
    We can approximate z_p using standard approximations for the upper tail.
    Or, simpler: use a lookup or a robust approximation for the lower tail.
    
    Actually, for this task, we can just hardcode a few z-values or use a simple approximation.
    Since alpha is dynamic but small, let's use an approximation for z_alpha.
    
    A common approximation for the inverse CDF in the tails is based on the Mills ratio or similar.
    However, for simplicity and speed, we can precompute a small set or use a generic formula.
    
    Let's use the approximation for the quantile function of the standard normal:
    For p in [0.5, 1], z_p = sqrt(-2 log(1-p)) * (c0 + c1 sqrt(log(1-p)) + ...) is complex.
    
    Alternative: Use the fact that for small alpha, z_alpha is roughly -sqrt(-2 log(alpha)) * (1 - 1/(4 log(alpha)) ...).
    
    Let's use a simpler, robust approach: 
    We don't strictly need high precision for the PPF in this heuristic context, 
    but we want it to be monotonic and reasonable.
    
    Let's use a standard approximation for the inverse error function if available, 
    or just a polynomial approximation for the Q-function inverse.
    
    Actually, scipy.stats.norm.ppf is available but might be slow if called many times? 
    No, it's called once per acquisition step (scalar alpha), not per vector element. 
    So we can use scipy.stats.norm.ppf if we import it. 
    But the prompt says "Python standard library, NumPy and SciPy are available".
    So I can use scipy.stats.norm.ppf.
    """
    from scipy.stats import norm
    return norm.ppf(p)

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    fb = float(f_best)

    # Check deadline
    if time.time() > _DEADLINE:
        return -mu

    s = np.maximum(sigma, 1e-8)

    # Strategy: Lower Quantile (LQ)
    # Score = mu - z_alpha * sigma
    # We want to find the point with the lowest potential value.
    # z_alpha corresponds to the alpha-quantile of the standard normal.
    # If alpha is small (e.g., 0.01), z_alpha is negative (approx -2.33).
    # Then -z_alpha * sigma is positive.
    # Wait, LQ = mu + z_alpha * sigma. Since z_alpha is negative, this is mu - |z_alpha| * sigma.
    # This is equivalent to LCB with beta = |z_alpha|.
    
    # The key difference from the previous "dynamic blend" is that we are NOT blending with EI.
    # We are purely optimizing for the lower tail of the posterior.
    # This is a "stochastic sampling" perspective: if we could sample many times,
    # the minimum expected value is related to the lower quantile.
    
    # Dynamic Alpha:
    # Early on (high f_best), we are uncertain and far from optimum. We might want to be more "optimistic" (lower alpha, higher beta) to explore.
    # Late on (low f_best), we want to be more "pessimistic" (higher alpha, lower beta) to exploit.
    
    # Let's map f_best to alpha.
    # f_best range: [0, 25].
    # If f_best = 25 (start), alpha = 1e-4 (very exploratory, beta ~ 3.7)
    # If f_best = 0 (end), alpha = 0.1 (exploitative, beta ~ 1.28)
    
    # Linear interpolation in log-space for alpha?
    # log_alpha_start = log(1e-4) = -9.21
    # log_alpha_end = log(0.1) = -2.30
    
    # progress = 1 - f_best / 25.0
    # Clamp progress to [0, 1]
    progress = 1.0 - min(max(fb / 25.0, 0.0), 1.0)
    
    # Interpolate log_alpha
    log_alpha_start = -9.2103  # log(1e-4)
    log_alpha_end = -2.3026    # log(0.1)
    
    log_alpha = log_alpha_start + progress * (log_alpha_end - log_alpha_start)
    alpha = np.exp(log_alpha)
    
    # Get z_alpha
    z_alpha = _norm_ppf(alpha)
    
    # Calculate scores
    # LQ = mu + z_alpha * sigma
    # Since we want "higher score means evaluate next", and LQ is "lower is better" for the function value,
    # we should return -LQ to make the minimum function value the maximum score.
    # Or, we can just return - (mu + z_alpha * sigma).
    
    scores = -(mu + z_alpha * s)
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
