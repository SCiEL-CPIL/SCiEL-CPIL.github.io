"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def _pareto_frontier(mu, sigma, cost, f_best):
    """
    Fast Pareto-frontier score using vectorized dominance counting.
    Points on the non-dominated frontier get high scores; dominated points
    get scores inversely proportional to how many points dominate them.
    """
    n = len(mu)
    
    # Objectives to maximize
    imp = f_best - mu        # predicted improvement
    unc = sigma.copy()       # uncertainty
    neg_cost = -cost.copy()  # negative cost (maximize = minimize cost)
    
    # Vectorized dominance counting
    imp_ge = imp[:, None] >= imp[None, :]
    unc_ge = unc[:, None] >= unc[None, :]
    nc_ge = neg_cost[:, None] >= neg_cost[None, :]
    
    imp_gt = imp[:, None] > imp[None, :]
    unc_gt = unc[:, None] > unc[None, :]
    nc_gt = neg_cost[:, None] > neg_cost[None, :]
    any_strict = imp_gt | unc_gt | nc_gt
    
    dominates = imp_ge & unc_ge & nc_ge & any_strict
    
    dom_count = dominates.sum(axis=0)
    
    pf_scores = np.exp(-0.5 * dom_count)
    pf_scores[dom_count == 0] = 1.0
    
    return pf_scores


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Combines Pareto-frontier selection with cost-aware UCB/EI blending.
    
    Parameters
    ----------
    mu : array (2000,) GP mean
    sigma : array (2000,) GP std
    f_best : float best observed value
    cost : array (2000,) evaluation cost per candidate
    
    Returns
    -------
    scores : array (2000,) higher = evaluate next
    """
    t0 = time.time()
    DEADLINE = 320.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1.0)
    
    n = len(mu)
    
    # --- Phase 1: Compute base acquisition scores ---
    
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-9:
        mu_range = 1.0
    
    sigma_median = np.median(sigma)
    sigma_rel = sigma_median / mu_range
    
    # Adaptive beta with cost-based progress
    beta_max = 4.5
    beta_min = 1.2
    beta_t = beta_min + (beta_max - beta_min) * np.clip(sigma_rel * 2.0, 0, 1)
    
    cost_mean = np.mean(cost)
    cost_factor = 1.0 / (1.0 + 0.12 * (cost_mean - 1.0))
    beta_t *= cost_factor
    
    beta_cost = 0.8
    ucb_term = -mu + beta_t * sigma
    cost_normalizer = cost ** beta_cost
    
    # EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Robust percentile normalization for EI
    ei_p95 = np.percentile(ei, 95)
    if ei_p95 > 1e-9:
        ei_norm = ei / ei_p95
        ei_norm = np.clip(ei_norm, 0, 1.0)
    else:
        ei_norm = np.zeros_like(ei)
    
    # --- Phase 2: Pareto frontier computation ---
    
    if time.time() - t0 > DEADLINE - 100:
        pf_scores = np.ones(n)
    else:
        pf_scores = _pareto_frontier(mu, sigma, cost, f_best)
    
    # --- Phase 3: Combine ---
    
    # Adaptive blending: early in search favor exploration, later favor exploitation
    ei_weight = np.clip(1.0 - 1.8 * sigma_rel, 0.25, 0.75)
    ucb_weight = 1.0 - ei_weight
    
    # Robust percentile normalization for UCB
    ucb_p95 = np.percentile(ucb_term, 95)
    ucb_p5 = np.percentile(ucb_term, 5)
    ucb_range = ucb_p95 - ucb_p5
    if ucb_range > 1e-9:
        ucb_norm = (ucb_term - ucb_p5) / ucb_range
        ucb_norm = np.clip(ucb_norm, 0, 1.0)
    else:
        ucb_norm = np.zeros_like(ucb_term)
    
    # Cost-aware blended score
    blended = ei_weight * ei_norm + ucb_weight * ucb_norm
    blended_cost = blended / cost_normalizer
    
    # Pareto boost: smooth log-sum-exp style boost for frontier points
    # Use a softer boost that doesn't over-concentrate
    pareto_boost = 0.35
    final_scores = blended_cost * (1.0 + pareto_boost * pf_scores)
    
    # Ensure finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    # Final safety: if we're close to deadline, just return what we have
    return final_scores
# EVOLVE-BLOCK-END
