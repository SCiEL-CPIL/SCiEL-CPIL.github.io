"""Direct cost-aware acquisition for Bayesian optimization of rosenbrock 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 325.0  # 5s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Adaptive blend of Probabilistic Improvement (exploration) and
    quantile-based exploitation, with cost normalization.
    
    Structural change: Instead of a single fixed acquisition (pure quantile),
    we compute two complementary scores and blend them with an adaptive weight
    that depends on the relationship between f_best and the predictive mean.
    
    - PI score: P(f(x) < f_best) = Phi((f_best - mu) / sigma) — favors exploration
      in uncertain regions where there's a chance of beating the current best.
    - Quantile score: -q_0.1 / cost — favors high-confidence low-cost exploitation.
    
    The blend weight shifts from exploration-heavy early (when f_best is still high
    and uncertain) to exploitation-heavy later (when we have a good best and want
    to refine).
    
    Both components are normalized to zero-mean, unit-variance before blending
    to ensure they contribute comparably regardless of scale.
    """
    # Deadline guard: if we're running out of time, return simple exploitation
    if time.time() > _DEADLINE:
        safe_cost = np.maximum(cost, 1e-6)
        scores = -mu / safe_cost
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle NaN/Inf
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Component 1: Probabilistic Improvement (PI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is high when mu is well below f_best (exploitation) or when sigma is
    # large relative to the gap (exploration).
    z_pi = (f_best - mu) / sigma
    # Use a stable approximation for Phi
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # For large positive z, Phi -> 1; for large negative z, Phi -> 0
    z_pi_clipped = np.clip(z_pi, -8.0, 8.0)
    pi_score = 0.5 * (1.0 + np.erf(z_pi_clipped / np.sqrt(2.0)))
    
    # Component 2: Cost-effective quantile (exploitation)
    # q_0.1 = mu - 1.2816 * sigma
    # Score: -q_0.1 / cost (higher is better)
    z_quantile = -1.2815515655446004
    q_0_1 = mu + z_quantile * sigma
    quantile_score = -q_0_1 / cost
    
    # Normalize both components to zero-mean, unit-variance
    def _normalize(arr):
        m = np.mean(arr)
        s = np.std(arr)
        if s < 1e-10:
            return np.zeros_like(arr)
        return (arr - m) / s
    
    pi_norm = _normalize(pi_score)
    quant_norm = _normalize(quantile_score)
    
    # Adaptive blend weight:
    # When f_best is high relative to the GP mean, we're early — explore more.
    # When f_best is low (close to GP min), we're late — exploit more.
    # Measure: how much does f_best exceed the GP mean?
    mu_mean = np.mean(mu)
    # If f_best >> mu_mean, we're early (high exploration weight)
    # If f_best <= mu_mean, we're late (high exploitation weight)
    gap = f_best - mu_mean
    # Map gap to weight in [0, 1]: weight for PI (exploration)
    # Use a sigmoid-like mapping centered around 0
    # When gap is large positive, w_pi -> 1 (explore)
    # When gap is negative or small, w_pi -> 0.3 (exploit more)
    # Scale: rosenbrock values can range from 0 to hundreds
    # Use a robust scale based on the spread of mu
    mu_spread = np.std(mu) + 1e-8
    # Normalize the gap by the spread of predictions
    normalized_gap = gap / (mu_spread + 1e-8)
    
    # Sigmoid mapping: w_pi = 0.3 + 0.7 * sigmoid(normalized_gap)
    # This ensures w_pi is always in [0.3, 1.0], so we never fully ignore exploration
    sig = 1.0 / (1.0 + np.exp(-np.clip(normalized_gap, -10.0, 10.0)))
    w_pi = 0.3 + 0.7 * sig
    w_quant = 1.0 - w_pi
    
    # Blend
    scores = w_pi * pi_norm + w_quant * quant_norm
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
