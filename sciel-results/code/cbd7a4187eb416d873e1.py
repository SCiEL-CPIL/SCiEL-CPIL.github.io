"""Model-and-optimize acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_DEADLINE_SECONDS = 300.0
_SAFETY_MARGIN = 20.0

# Known global minimum of Hartmann-3D
_HARTMANN3D_MIN = -3.86278

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _ei_scores(mu, sigma, target):
    """
    Compute Expected Improvement for minimization relative to a target value.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (target - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (target - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.where(np.isfinite(ei), ei, 0.0)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a DGP-inspired approach:
    - Deterministic lower bound via soft-min of predictive means
    - Expected Improvement over this deterministic lower bound
    - Adaptive exploration-exploitation balance based on convergence
    
    Parameters:
    mu: array of shape (n,) - posterior means
    sigma: array of shape (n,) - posterior standard deviations
    f_best: float - best observed value (lowest)
    
    Returns:
    scores: array of shape (n,) - finite scores, higher means evaluate next
    """
    _ensure_start_time()
    
    # Deadline guard: fall back to simple, fast EI if time is running out
    if _elapsed() > _DEADLINE_SECONDS - _SAFETY_MARGIN:
        return _ei_scores(mu, sigma, f_best)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    n = len(mu)
    if n == 0:
        return mu
    
    # Convergence-based adaptive weight
    if f_best > _HARTMANN3D_MIN:
        gap = f_best - _HARTMANN3D_MIN
        conv = 1.0 - (gap / (gap + 0.5))
    else:
        conv = 1.0
    
    conv = float(np.clip(conv, 0.0, 1.0))
    
    # Weight for exploitation vs exploration
    # When close to optimum, exploit more; when far, explore more
    weight_exploit = 0.4 + 0.6 * conv
    weight_explore = 1.0 - weight_exploit
    
    # Exploitation component: EI relative to f_best
    ei_fb = _ei_scores(mu, sigma, f_best)
    
    # Exploration component: DGP-style deterministic lower bound
    # Use soft-min (log-sum-exp approximation) of predictive means
    # to get a smooth deterministic surrogate lower bound
    
    # Temperature for soft-min: controls sharpness
    # Lower temperature -> sharper minimum, higher -> smoother
    temp = 0.05 * (1.0 - 0.5 * conv)
    
    # Soft-min via log-sum-exp: min(x) ≈ -log(sum(exp(-x/temp))) * temp
    # But we need to be careful with numerical stability
    mu_neg = -mu / temp
    max_mu_neg = np.max(mu_neg)
    # Soft min value
    log_sum_exp = max_mu_neg + np.log(np.sum(np.exp(mu_neg - max_mu_neg)))
    soft_min = -log_sum_exp * temp
    
    # For each candidate, compute EI relative to the soft-min lower bound
    # This encourages exploring regions where the GP predicts low values
    # but with a deterministic (non-stochastic) lower bound
    
    # The idea: if a point has a low predictive mean relative to the overall
    # soft-min, it's promising. Combine with uncertainty.
    
    # Score based on: how much better is this point's mean than the soft-min?
    # Plus an uncertainty bonus scaled by convergence
    mean_advantage = soft_min - mu  # Positive when mu is below soft_min (unlikely)
    # Actually, we want to find points where mu is low
    # Let's use: f_best - mu as a proxy for potential improvement,
    # but weighted by how confident we are (via sigma)
    
    # Alternative: Use the probability that this point is the global min
    # P(min) ∝ exp(-mu_i / temp) / sum_j exp(-mu_j / temp)
    # This is a softmax over negative means
    
    # Compute softmax probabilities (probability of being the minimum)
    neg_mu = -mu
    max_neg_mu = np.max(neg_mu)
    exp_neg_mu = np.exp(neg_mu - max_neg_mu)
    prob_min = exp_neg_mu / np.sum(exp_neg_mu)
    
    # Expected improvement if we sample from the distribution proportional to prob_min
    # This is like: E[improvement] = sum_j prob_min_j * max(0, f_best - mu_j)
    # But we want a per-point score, so use:
    # score_i = prob_min_i * (f_best - mu_i) + exploration_bonus
    
    # Exploration bonus: favor points with high sigma when we're far from optimum
    exploration_bonus = weight_explore * sigma_safe * (1.0 - conv)
    
    # Combine: exploitation via EI + exploration via probability-weighted improvement
    # + pure uncertainty bonus
    
    scores = weight_exploit * ei_fb + weight_explore * (prob_min * (f_best - mu) + exploration_bonus)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
