import numpy as np
import time

# Deadline guard: the evaluator kills after 330s. We track time and if we exceed
# a safety margin, we return a safe fallback.
_START_TIME = time.time()
_TIME_LIMIT = 325.0  # 5s safety margin before 330s

def _check_time():
    return (time.time() - _START_TIME) > _TIME_LIMIT

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a Pareto-frontier filtering strategy: first masks out candidates whose
    predicted mean is worse than f_best by more than a tolerance, then applies
    a pure UCB score to the remaining candidates to aggressively exploit
    high-uncertainty regions with plausible improvement potential.
    """
    # Check deadline
    if _check_time():
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        # Fallback: pick the point with lowest mu (most exploitative)
        return -mu.astype(np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Floor sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Pareto-frontier filtering:
    # Mask out candidates whose predicted mean is worse than f_best by more than a tolerance.
    # The tolerance scales with the current uncertainty level to adaptively define "plausible improvement".
    # Use a tolerance that is a fraction of the typical sigma scale.
    sigma_mean = np.mean(sigma_safe)
    # Tolerance: allow candidates that are within a small multiple of sigma_mean from f_best
    # This ensures we keep candidates that could plausibly beat f_best
    tolerance = 0.5 * sigma_mean + 1e-6
    
    # A candidate is "plausible" if mu <= f_best + tolerance
    plausible_mask = mu <= (f_best + tolerance)
    
    # If no candidates pass the filter (shouldn't happen often), fall back to all
    if not np.any(plausible_mask):
        plausible_mask = np.ones(n, dtype=bool)
    
    # Pure UCB score for minimization: -mu + beta * sigma
    # Beta is adaptive: higher when far from optimum (f_best large), lower when close
    f_scale = 0.5
    f_norm = min(max(f_best, 0.0), f_scale) / f_scale  # in [0, 1]
    
    # Beta ranges from ~1.0 (exploitative) to ~3.0 (exploratory)
    beta = 1.0 + 2.0 * f_norm
    
    # Compute UCB score for all candidates
    ucb_score = -mu + beta * sigma_safe
    
    # Apply Pareto filter: set non-plausible candidates to very low scores
    # This effectively restricts the search to regions with plausible improvement
    score = np.full(n, -1e10, dtype=np.float64)
    score[plausible_mask] = ucb_score[plausible_mask]
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
