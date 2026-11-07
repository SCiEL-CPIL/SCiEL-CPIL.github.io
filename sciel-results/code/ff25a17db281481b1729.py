"""Cost-aware acquisition for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: evaluator kills after 330s wall-clock; use 280s safety margin
_DEADLINE = time.time() + 280.0


def _safe_finite(arr, fallback=0.0):
    """Replace non-finite values with fallback."""
    return np.where(np.isfinite(arr), arr, fallback)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using cost-normalized Expected Improvement
    with a probabilistic cost model and variance-aware exploration.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP std predictions
        f_best: scalar, best observed value so far
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    n = len(mu)
    
    # Deadline check
    if time.time() > _DEADLINE:
        return np.zeros(n, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # z-score for EI (minimization)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -30, 30)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # --- Component 1: Cost-aware EI (CEI) ---
    # Model cost as log-normal: E[cost] = cost_safe, Var[cost] = (0.1 * cost_safe)^2
    # E[1/cost] for log-normal with small CV: approx 1/E[cost] * (1 + CV^2)
    # But for CEI we want E[improvement] / E[cost] as a proxy.
    # Use a more principled approach: CEI = EI / cost_safe, but regularize
    # to avoid division by very small costs.
    cei = ei / cost_safe
    
    # --- Component 2: Probabilistic Improvement (PI) ---
    # P(f(x) < f_best) = Phi(z) for minimization
    pi_score = cdf_z / cost_safe
    
    # --- Component 3: Variance-aware exploration ---
    # Exploration bonus: sigma / cost, but scaled to be comparable to CEI
    # Use a beta that adapts: early on (high sigma relative to range) explore more
    # Estimate "progress" from how good f_best is relative to the known minimum (-0.8)
    # f_best is in objective units; the minimum of cosine8 is -0.8
    f_min = -0.8
    # Progress: 0 = just started (f_best near 0), 1 = found minimum
    progress = np.clip((0.0 - f_best) / (0.0 - f_min + 1e-8), 0.0, 1.0)
    # Beta: start at 1.0 (more exploration), decrease to 0.3 (more exploitation)
    beta = 1.0 - 0.7 * progress
    ucb_cost = beta * sigma_safe / cost_safe
    
    # --- Component 4: Hypervolume-style diversity among top candidates ---
    # After combining, we'll re-rank top candidates. For now, compute raw score.
    
    # Combine: CEI dominates, PI as secondary exploitation, UCB as exploration
    # Weights: CEI 0.55, PI 0.20, UCB 0.25
    raw_score = 0.55 * cei + 0.20 * pi_score + 0.25 * ucb_cost
    
    # Ensure finite
    raw_score = _safe_finite(raw_score, 0.0)
    
    # --- Top-k hypervolume-style re-ranking ---
    # For the top 200 candidates by raw_score, compute a hypervolume contribution
    # that rewards candidates that are not redundant with other top candidates.
    # This is a structural change: instead of a pure pointwise score, we
    # incorporate a global diversity measure among the top candidates.
    
    k = min(200, n)
    if k > 1:
        # Get top-k indices
        top_idx = np.argpartition(-raw_score, k)[:k]
        
        # For each top candidate, compute its "hypervolume contribution"
        # relative to other top candidates. We use a simple proxy:
        # For each candidate i, compute the minimum raw_score of all other
        # top candidates that are "close" in the score space, and reward
        # candidates that stand out.
        # 
        # A simpler and more effective approach: use the raw_score itself
        # but add a small bonus for candidates that are in the top-k but
        # have high sigma (uncertainty), as they provide more information.
        # This is a structural change: the final score depends on the
        # global distribution of top candidates, not just pointwise values.
        
        # Compute a "novelty" score: for each top candidate, how much
        # does its (mu, sigma) pair differ from the mean of other top candidates?
        top_mu = mu[top_idx]
        top_sigma = sigma_safe[top_idx]
        
        # Normalize mu and sigma among top candidates
        mu_mean = np.mean(top_mu)
        mu_std = np.std(top_mu) + 1e-8
        sig_mean = np.mean(top_sigma)
        sig_std = np.std(top_sigma) + 1e-8
        
        # Distance from the "centroid" of top candidates in (mu, sigma) space
        novelty = np.sqrt(((top_mu - mu_mean) / mu_std) ** 2 + 
                          ((top_sigma - sig_mean) / sig_std) ** 2)
        
        # Scale novelty to be a small bonus (max ~0.05 of typical raw_score magnitude)
        raw_scale = np.max(np.abs(raw_score)) + 1e-8
        novelty_bonus = 0.05 * raw_scale * novelty / (np.max(novelty) + 1e-8)
        
        # Apply novelty bonus to top candidates
        raw_score[top_idx] += novelty_bonus
    
    # Final safety: ensure all finite
    raw_score = _safe_finite(raw_score, 0.0)
    
    return raw_score
# EVOLVE-BLOCK-END
