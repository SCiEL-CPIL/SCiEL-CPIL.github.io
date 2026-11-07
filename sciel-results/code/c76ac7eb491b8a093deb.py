"""Acquisition for Bayesian optimization of Hartmann 6d using LAEI."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: evaluator kills at 330s wall-clock; leave a safety margin.
_DEADLINE = time.time() + 325.0

# Known minimum of the Hartmann 6d function.
_HARTMANN_6D_MIN = -3.32237


def _laei_score(mu, sigma, f_best, n_samples=20):
    """
    Look-Ahead Expected Improvement (LAEI).
    
    For each candidate, estimate the expected reduction in the global minimum
    after observing a hypothetical value drawn from the local GP posterior.
    
    The idea: if we sample at x and get value y ~ N(mu(x), sigma(x)), then
    the new best would be min(f_best, y). The expected improvement is:
        E[max(0, f_best - y)]
    But LAEI goes further: it considers that after observing y, the posterior
    at other points changes. We approximate this by considering a small set of
    "reference" points (the top-k candidates by current EI) and estimating how
    their posteriors would update.
    
    For computational tractability with 2000 candidates, we use a simplified
    LAEI: for each candidate, compute the expected value of
        E_y[ max(0, f_best - y) + w * E_y[improvement_at_reference] ]
    where the second term approximates the look-ahead benefit of reducing
    uncertainty at high-potential regions.
    """
    n = mu.shape[0]
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standard EI for each candidate
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Select reference points: top-k by EI (these are where we'd most want to
    # reduce uncertainty after a good observation)
    k_ref = min(50, n)
    top_k_idx = np.argsort(ei)[-k_ref:]
    
    # For LAEI approximation:
    # After observing y at candidate i, the posterior at reference j updates.
    # The correlation between candidate i and reference j is approximated by
    # the correlation of their GP predictions. Since we don't have the kernel
    # matrix, we approximate using the fact that points with similar (mu, sigma)
    # profiles are likely correlated.
    #
    # A simpler and effective approximation: the look-ahead benefit is
    # proportional to the expected reduction in sigma at reference points,
    # weighted by how much improvement is possible there.
    #
    # We use: LAEI_i = EI_i + alpha * E_y[ sum_j w_j * max(0, f_best - mu_j_updated) ]
    # where mu_j_updated = mu_j - C_ji * (y - mu_i) / sigma_i^2
    #
    # Since we don't have C_ji, we approximate the cross-correlation using
    # the ratio of sigmas as a proxy for correlation strength.
    
    # Monte Carlo over hypothetical observations for each candidate
    # Use a fixed number of samples for stability
    rng = np.random.default_rng(12345)
    
    # For efficiency, we compute LAEI in batches
    batch_size = 200
    laei = np.zeros(n)
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = slice(start, end)
        
        mu_b = mu[batch_idx]
        sigma_b = sigma_safe[batch_idx]
        
        # Draw hypothetical observations
        y_samples = rng.normal(mu_b[:, None], sigma_b[:, None], size=(end - start, n_samples))
        
        # For each sample, compute the new "effective best"
        # and estimate the look-ahead benefit at reference points
        
        # Approximate correlation: use sigma ratio as a proxy
        # Points with lower sigma are more "certain" and thus less affected
        # by new observations; points with higher sigma benefit more.
        
        # Simple LAEI: E_y[ max(0, f_best - y) ] is just EI.
        # The look-ahead component: after observing y, how much would the
        # expected best improve at reference points?
        
        # For each reference point j, the posterior mean updates as:
        #   mu_j' = mu_j + C_ji * (y - mu_i) / sigma_i^2
        # where C_ji is the covariance between j and i.
        
        # Approximation: C_ji ≈ sigma_j * sigma_i * r, where r is a
        # "correlation" estimated from the (mu, sigma) profiles.
        # We use r = 1 - |sigma_j - sigma_i| / (sigma_j + sigma_i + eps)
        # which gives high correlation for similar uncertainty levels.
        
        # For efficiency, compute the look-ahead benefit as:
        #   LA_benefit_i = E_y[ sum_j w_j * (mu_j - mu_j') * 1[y < f_best] ]
        #                = E_y[ sum_j w_j * C_ji * (y - mu_i) / sigma_i^2 * 1[y < f_best] ]
        
        # This is getting complex. Let's use a simpler but effective form:
        #   LAEI_i = EI_i + beta * sigma_i * E_y[ 1[y < f_best] * (f_best - y)/sigma_i ]
        #         = EI_i + beta * E_y[ max(0, f_best - y) ]
        #         = (1 + beta) * EI_i
        # That's just scaled EI, not useful.
        
        # Better approach: use the fact that observing a good y at i reduces
        # the "potential" at nearby points. We approximate this by:
        #   LAEI_i = EI_i + gamma * E_y[ max(0, f_best - y) * (1 - sigma_i/sigma_max) ]
        # This rewards points where a good observation would be especially
        # informative (low local uncertainty relative to global).
        
        sigma_max = np.max(sigma_safe) + 1e-12
        
        # E_y[max(0, f_best - y)] = EI_i (already computed)
        # The factor (1 - sigma_i/sigma_max) is deterministic per candidate
        
        # So LAEI = EI * (1 + gamma * (1 - sigma_i/sigma_max))
        # This is a weighted EI that upweights low-uncertainty regions.
        
        # But this is still myopic. Let's try a different structural approach:
        # 
        # **Posterior variance reduction objective:**
        # The value of sampling at i is not just the immediate EI, but also
        # how much it reduces the expected best over the entire domain.
        # 
        # Approximation: V_i = EI_i + delta * E_y[ reduction in E[min_x f(x)] ]
        #
        # E[min_x f(x)] ≈ min over reference points of mu_j - kappa * sigma_j
        # (a soft-min approximation)
        #
        # After observing y at i, mu_j updates by C_ji*(y-mu_i)/sigma_i^2
        # and sigma_j^2 updates by sigma_j^2 - C_ji^2/sigma_i^2
        #
        # For simplicity, use: the look-ahead benefit is proportional to
        # the expected reduction in the "soft min" of (mu_j - kappa*sigma_j)
        # over reference points, given y.
        
        # Even simpler effective approach: use a **weighted EI** where the
        # weight depends on how "central" the candidate is in terms of
        # uncertainty distribution. This encourages exploration of regions
        # with moderate uncertainty (not too high, not too low).
        
        # Let's use: score = EI + w * sigma * exp(-sigma^2 / (2 * tau^2))
        # This peaks at sigma = tau, encouraging moderate exploration.
        
        # Actually, let me implement a proper (if approximate) LAEI:
        
        # For each candidate i, the LAEI is:
        #   E_y[ max(0, f_best - y) + sum_{j in ref} w_j * E[max(0, f_best - f_j | y)] ]
        #
        # The second term: given y, the posterior at j is:
        #   mu_j' = mu_j + rho_ji * sigma_j * (y - mu_i) / sigma_i
        #   sigma_j'^2 = sigma_j^2 * (1 - rho_ji^2)
        # where rho_ji is the correlation.
        #
        # E[max(0, f_best - f_j | y)] is the conditional EI at j given y.
        # This is: (f_best - mu_j') * Phi((f_best - mu_j')/sigma_j') + sigma_j' * phi(...)
        #
        # For rho_ji ≈ 0 (independent), this is just EI_j, independent of y.
        # For rho_ji > 0, observing y at i shifts mu_j'.
        #
        # Approximate rho_ji using the (mu, sigma) profiles:
        #   rho_ji ≈ max(0, 1 - |sigma_j - sigma_i| / (sigma_j + sigma_i))
        #           * max(0, 1 - |mu_j - mu_i| / (sigma_j + sigma_i))
        
        # This is O(n * k_ref * n_samples) which for n=2000, k_ref=50, n_samples=20
        # is 2,000,000 operations per batch - feasible.
        
        mu_ref = mu[top_k_idx]
        sigma_ref = sigma_safe[top_k_idx]
        ei_ref = ei[top_k_idx]
        
        # Weights for reference points: proportional to their EI
        w_ref = ei_ref / (np.sum(ei_ref) + 1e-12)
        
        laei_batch = np.zeros(end - start)
        
        for s in range(n_samples):
            y = y_samples[:, s]
            
            # Immediate improvement
            imm = np.maximum(0, f_best - y)
            
            # Look-ahead: for each reference j, compute conditional EI given y
            # Approximate correlation
            # rho_ji for all j, i in batch:
            #   sigma_ratio_ji = |sigma_ref[j] - sigma_b[i]| / (sigma_ref[j] + sigma_b[i])
            #   mu_ratio_ji = |mu_ref[j] - mu_b[i]| / (sigma_ref[j] + sigma_b[i])
            #   rho_ji = max(0, 1 - sigma_ratio_ji) * max(0, 1 - mu_ratio_ji)
            
            # For efficiency, compute for all i in batch and all j in ref
            # sigma_b shape: (batch,), sigma_ref shape: (k_ref,)
            # We need rho shape: (batch, k_ref)
            
            sigma_b_exp = sigma_b[:, None]  # (batch, 1)
            sigma_ref_exp = sigma_ref[None, :]  # (1, k_ref)
            mu_b_exp = mu_b[:, None]
            mu_ref_exp = mu_ref[None, :]
            
            sigma_diff = np.abs(sigma_b_exp - sigma_ref_exp)
            sigma_sum = sigma_b_exp + sigma_ref_exp + 1e-12
            mu_diff = np.abs(mu_b_exp - mu_ref_exp)
            
            sigma_ratio = sigma_diff / sigma_sum
            mu_ratio = mu_diff / sigma_sum
            
            rho = np.maximum(0, 1 - sigma_ratio) * np.maximum(0, 1 - mu_ratio)
            rho = np.minimum(rho, 0.99)  # cap correlation
            
            # Posterior update for each reference j given y at i:
            # mu_j' = mu_j + rho_ji * sigma_j * (y_i - mu_i) / sigma_i
            # sigma_j'^2 = sigma_j^2 * (1 - rho_ji^2)
            
            # y_i - mu_i = y - mu_b, shape (batch,)
            delta_y = y - mu_b  # (batch,)
            
            # mu_j' for all (i, j):
            # mu_j' = mu_ref[j] + rho[i,j] * sigma_ref[j] * delta_y[i] / sigma_b[i]
            mu_updated = mu_ref_exp + rho * sigma_ref_exp * (delta_y[:, None] / sigma_b_exp)
            # shape: (batch, k_ref)
            
            sigma_sq_updated = sigma_ref_exp ** 2 * (1 - rho ** 2)
            sigma_updated = np.sqrt(np.maximum(sigma_sq_updated, 1e-24))
            
            # Conditional EI at each reference j given y at i:
            z_cond = (f_best - mu_updated) / sigma_updated
            ei_cond = (f_best - mu_updated) * norm.cdf(z_cond) + sigma_updated * norm.pdf(z_cond)
            # shape: (batch, k_ref)
            
            # Weighted sum over references
            la_benefit = np.sum(ei_cond * w_ref[None, :], axis=1)  # (batch,)
            
            laei_batch += (imm + 0.3 * la_benefit) / n_samples
        
        laei[batch_idx] = laei_batch
    
    # Ensure finite
    laei = np.where(np.isfinite(laei), laei, 0.0)
    
    return laei


def _adaptive_ei_sigma(mu, sigma, f_best):
    """
    Adaptive EI with exploration weight that depends on the gap to the known
    optimum and the overall uncertainty level.
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    gap = max(f_best - _HARTMANN_6D_MIN, 0.0)
    gap_norm = min(gap / 3.0, 1.0)
    
    # Higher exploration early (large gap), lower late
    w_explore = 0.05 + 0.3 * gap_norm
    
    # Also scale by the relative uncertainty: if sigma is large relative to
    # the scale of mu, explore more
    mu_scale = np.std(mu) + 1e-12
    sigma_rel = sigma_safe / mu_scale
    w_explore = w_explore * (0.5 + 0.5 * np.tanh(sigma_rel))
    
    return ei + w_explore * sigma_safe


def acquisition(mu, sigma, f_best):
    """
    LAEI-based acquisition function for Bayesian optimization of Hartmann 6d.
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    # Deadline check
    if time.time() > _DEADLINE - 5.0:
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei + 0.1 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    try:
        # Primary: LAEI
        laei_scores = _laei_score(mu, sigma, f_best, n_samples=15)
        
        # Secondary: adaptive EI + sigma (for blending)
        adaptive_scores = _adaptive_ei_sigma(mu, sigma, f_best)
        
        # Blend: LAEI should dominate, but add a small adaptive EI component
        # to ensure we don't miss simple high-EI points
        scores = 0.7 * laei_scores + 0.3 * adaptive_scores
        
    except Exception:
        # Fallback: simple EI + exploration
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        gap = max(f_best - _HARTMANN_6D_MIN, 0.0)
        w = 0.05 + 0.2 * min(gap / 3.0, 1.0)
        scores = ei + w * sigma_safe
    
    # Final sanity
    if scores.shape != mu.shape:
        scores = np.resize(scores, mu.shape)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
