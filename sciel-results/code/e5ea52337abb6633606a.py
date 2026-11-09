# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Structural change: replaces the internal grid-search calibration with an
    analytic, uncertainty-driven cost-aware LCB (Lower Confidence Bound) score.
    
    The score is:
        score(x) = (f_best - mu(x) + beta(t) * sigma(x)) / (cost(x) ** beta_c)
    
    where beta(t) adapts to the current posterior uncertainty relative to the
    function spread, and beta_c is a fixed cost sensitivity exponent.
    
    Key structural differences from the parent:
    1. No internal calibration loop (removes the dominant time cost and
       overfitting-to-proxy risk).
    2. LCB-based score directly targets the minimum rather than improvement
       over f_best, which can be more robust when f_best is not yet close
       to the true minimum.
    3. Adaptive exploration constant beta(t) scales with the ratio of mean
       sigma to the range of mu, encouraging exploration when the GP is
       uncertain and exploitation when it is confident.
    4. Fixed cost exponent beta_c = 1.0 (linear cost penalty) for stability.
    """
    t_start = time.time()
    TIME_LIMIT = 320.0  # Safety margin before 330s kill

    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-8)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    n = len(mu_safe)

    # --- Analytic adaptive LCB computation ---

    # Compute statistics of the current posterior
    mu_min = np.min(mu_safe)
    mu_max = np.max(mu_safe)
    mu_range = mu_max - mu_min + 1e-10
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)

    # Adaptive exploration parameter:
    # When uncertainty is high relative to the function range, explore more.
    # beta scales with the ratio of mean sigma to the function range.
    # We also add a base exploration to avoid pure exploitation.
    uncertainty_ratio = sigma_mean / (mu_range + 1e-10)

    # Base exploration constant, scaled by uncertainty
    # Higher uncertainty -> more exploration
    base_beta = 1.0
    adaptive_factor = 1.0 + 2.0 * uncertainty_ratio

    # The effective beta: a single scalar that adapts globally
    beta = base_beta * adaptive_factor

    # Cap beta to prevent extreme exploration
    beta = np.clip(beta, 0.5, 5.0)

    # Cost exponent: fixed at 1.0 for stability (linear cost penalty)
    beta_c = 1.0

    # LCB score: we want to maximize (f_best - mu + beta * sigma) / cost^beta_c
    # This is equivalent to minimizing (mu - beta * sigma) * cost^beta_c / (f_best - mu + beta * sigma)
    # But more directly: score = improvement + exploration, normalized by cost
    lcb_improvement = f_best - mu_safe + beta * sigma_safe
    # Ensure improvement is non-negative (LCB should not be worse than f_best by much)
    # Actually, LCB can be worse than f_best if sigma is large, but that's okay for exploration
    # We don't clip to zero because exploration is valuable
    scores = lcb_improvement / (cost_safe ** beta_c)

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Time check: if we're running out of time, return immediately
    if time.time() - t_start > TIME_LIMIT:
        return scores

    # --- Optional refinement: boost high-uncertainty points among top candidates ---
    elapsed = time.time() - t_start
    if elapsed < TIME_LIMIT - 5.0:
        # Identify top 50 candidates
        top_k = min(50, n)
        top_idx = np.argsort(scores)[-top_k:]
        
        # For these top candidates, check if they are in high-uncertainty regions
        # and give them a small boost if so
        top_sigma = sigma_safe[top_idx]
        sigma_percentile = np.percentile(sigma_safe, 75)
        high_uncertainty = top_sigma > sigma_percentile
        
        # Small boost for high-uncertainty points among the top candidates
        # This encourages exploration in uncertain regions that already look promising
        boost = 0.1 * np.max(scores)
        for idx in top_idx:
            if high_uncertainty[np.where(top_idx == idx)[0][0]]:
                scores[idx] += boost

    # Final safety check
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
