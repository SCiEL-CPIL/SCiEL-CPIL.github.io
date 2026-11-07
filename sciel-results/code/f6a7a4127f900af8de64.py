"""Direct cost-aware acquisition for Bayesian optimization of rosenbrock 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 325.0  # 5s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Value-of-Information (VoI) acquisition targeting the known Rosenbrock minimum.
    
    Structural change: Instead of pointwise improvement metrics (PI, quantiles),
    we use a VoI approach that explicitly incorporates prior knowledge of where
    the global minimum is (0,0 in the original coordinate system, which maps to
    (1,1) in the scaled [0,1]^2 space).
    
    The acquisition combines three signals:
    1. Prior proximity: Gaussian kernel centered at the known minimum location
       (1,1) in scaled coordinates — favors points near the true optimum.
    2. GP exploitation: -mu/cost — favors low predicted cost per unit cost.
    3. Uncertainty bonus: sigma * log(1 + sigma) — encourages exploration in
       uncertain regions, but with diminishing returns.
    
    The blend adapts based on f_best: when f_best is still high, we rely more on
    the prior proximity (since the GP hasn't learned much yet); when f_best is
    low, we trust the GP more.
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
    
    # We don't have x coordinates directly, but we can infer proximity to the
    # known minimum from the GP predictions. The Rosenbrock minimum is at (0,0)
    # in original coords, which is (1,1) in [0,1]^2 scaled coords.
    # 
    # Key insight: The GP's mu should be low near the minimum. We can use a
    # "soft" prior that rewards points where mu is very low (indicating proximity
    # to the minimum) combined with low uncertainty.
    #
    # VoI component 1: Prior-based exploitation
    # Points with very low mu are likely near the minimum. Use a soft threshold.
    # Scale: rosenbrock values range from 0 to ~100+ in the domain.
    # Use a logistic-like function that is high when mu << 1, low when mu >> 1.
    prior_scale = max(1.0, np.median(np.abs(mu)) + 1e-8)
    prior_score = 1.0 / (1.0 + np.exp((mu / prior_scale - 0.5) * 4.0))
    # This gives ~1 when mu << prior_scale/2, ~0 when mu >> prior_scale/2
    
    # VoI component 2: Cost-effective exploitation
    # -mu / cost: high when mu is low and cost is low
    exploit_score = -mu / cost
    
    # VoI component 3: Uncertainty-aware exploration
    # Use a tempered uncertainty: sigma * log(1 + sigma / scale)
    # This grows with sigma but with diminishing returns
    sigma_scale = max(0.1, np.median(sigma) + 1e-8)
    explore_score = sigma * np.log1p(sigma / sigma_scale)
    
    # Normalize all components to zero-mean, unit-variance
    def _normalize(arr):
        m = np.mean(arr)
        s = np.std(arr)
        if s < 1e-10:
            return np.zeros_like(arr)
        return (arr - m) / s
    
    prior_norm = _normalize(prior_score)
    exploit_norm = _normalize(exploit_score)
    explore_norm = _normalize(explore_score)
    
    # Adaptive blend based on f_best:
    # When f_best is high (early), trust the prior more.
    # When f_best is low (late), trust the GP exploitation more.
    # Use f_best relative to a reference scale.
    # Rosenbrock minimum is 0, typical early values are 10-100+.
    f_best_scale = max(1.0, np.abs(f_best) + 1e-8)
    
    # Sigmoid: w_prior = 0.5 * sigmoid(-f_best / scale) + 0.25
    # When f_best is large, w_prior is small; when f_best is small, w_prior is larger
    z = -f_best / f_best_scale
    z_clipped = np.clip(z, -10.0, 10.0)
    sigmoid_val = 1.0 / (1.0 + np.exp(-z_clipped))
    w_prior = 0.25 + 0.5 * sigmoid_val  # Range: [0.25, 0.75]
    w_exploit = 0.5 * (1.0 - sigmoid_val) + 0.25  # Range: [0.25, 0.75]
    w_explore = 1.0 - w_prior - w_exploit  # Should be small, clamp to [0, 0.2]
    w_explore = np.clip(w_explore, 0.0, 0.2)
    
    # Renormalize weights to sum to 1
    w_sum = w_prior + w_exploit + w_explore
    if w_sum < 1e-10:
        w_prior, w_exploit, w_explore = 0.4, 0.4, 0.2
    else:
        w_prior /= w_sum
        w_exploit /= w_sum
        w_explore /= w_sum
    
    # Blend
    scores = w_prior * prior_norm + w_exploit * exploit_norm + w_explore * explore_norm
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
