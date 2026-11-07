"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Combined Weighted Expected Improvement + UCB acquisition.
    
    WEI: Expected improvement weighted by posterior probability of being the global minimum.
    UCB: Upper confidence bound for exploration.
    
    mu, sigma: shape (2000,) - GP mean and std in objective units
    f_best: lowest observed value
    Returns: 2000 finite scores, higher = evaluate next
    """
    # Guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # --- Expected Improvement (EI) ---
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # --- Weighted EI: weight by probability this point is the global min ---
    # P(point is global min) ≈ P(f(x) < f_best) = Φ((f_best - mu)/sigma)
    # This naturally down-weights points where we're already confident f_best is lower
    p_improve = norm.cdf(z)
    wei = ei * (1.0 + p_improve)  # Enhanced exploitation signal
    
    # --- UCB for exploration ---
    # Beta balances exploitation vs exploration
    # For 1D Weierstrass with ~25 evals, moderate exploration is needed
    beta = 1.5
    ucb = mu + beta * sigma_safe  # Higher mu (lower objective) and higher uncertainty
    
    # Note: We're minimizing, so lower mu is better.
    # UCB for minimization: we want low mu + high uncertainty
    # But acquisition should be HIGHER = more attractive
    # So for minimization: score = -mu + beta * sigma (exploit low values, explore uncertain)
    # Actually, let's think carefully:
    # - mu is the predicted objective (lower is better)
    # - We want to pick points with low mu (good predicted value) and/or high sigma (uncertain)
    # - EI already handles this: high when mu << f_best (improvement likely) or sigma large
    # - For UCB in minimization: score = -mu + beta * sigma
    #   This rewards low mu (negative of low = high score) and high sigma
    
    ucb_score = -mu + beta * sigma_safe
    
    # Combine: weighted sum of WEI and UCB
    # Normalize both to [0,1] range for stable combination
    # Use soft normalization to avoid division by zero
    wei_min = np.min(wei)
    wei_max = np.max(wei)
    wei_range = wei_max - wei_min + 1e-9
    wei_norm = (wei - wei_min) / wei_range
    
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # Weight: 70% exploitation (WEI), 30% exploration (UCB)
    # This favors finding good values while still exploring
    score = 0.7 * wei_norm + 0.3 * ucb_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
