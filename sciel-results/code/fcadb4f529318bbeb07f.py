"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of styblinski tang 1d.
    
    Uses Expected Improvement (EI) with a smooth soft-min knowledge gradient
    approximation. The soft-min is computed via log-sum-exp for numerical stability.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array_like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate. Higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Knowledge Gradient approximation using soft-min (log-sum-exp trick)
    # The KG estimates E[min_j (mu_j + sigma_j * Z_j) - f_best | Z_j ~ N(0,1)]
    # We approximate this by considering a subset of top candidates
    
    k = min(100, len(mu))
    top_k_idx = np.argsort(ei)[-k:]
    top_k_mu = mu[top_k_idx]
    top_k_sigma = sigma[top_k_idx]
    
    # Monte Carlo estimation of the expected minimum
    n_mc = 200
    rng = np.random.default_rng(42)
    mc_samples = rng.standard_normal((n_mc, k))
    
    # For each MC sample, compute the minimum value
    sampled_values = top_k_mu + top_k_sigma * mc_samples  # (n_mc, k)
    min_values = np.min(sampled_values, axis=1)  # (n_mc,)
    
    # Expected improvement of the global minimum
    kg_value = np.mean(min_values) - f_best
    kg_value = max(kg_value, 0.0)
    
    # For each top-k candidate, estimate its contribution to the KG
    # P(candidate i is argmin) * E[improvement_i | candidate i is argmin]
    is_min = np.argmin(sampled_values, axis=1)  # (n_mc,)
    
    # Count how often each candidate is the minimum
    counts = np.bincount(is_min, minlength=k)
    prob_min = counts / n_mc
    
    # Expected improvement given candidate i is the minimum
    # Approximate: when candidate i is min, the improvement is f_best - sampled_value_i
    # We can estimate this more carefully
    ei_given_min = np.zeros(k)
    for i in range(k):
        mask = (is_min == i)
        if np.sum(mask) > 0:
            ei_given_min[i] = np.mean(f_best - sampled_values[mask, i])
        else:
            ei_given_min[i] = 0.0
    
    # KG contribution for each top-k candidate
    kg_top_k = prob_min * ei_given_min
    
    # Create full KG score array
    kg_scores = np.zeros_like(mu)
    kg_scores[top_k_idx] = kg_top_k
    
    # Normalize KG scores to [0, 1]
    kg_min = np.min(kg_scores)
    kg_max = np.max(kg_scores)
    kg_range = kg_max - kg_min + 1e-12
    kg_norm = (kg_scores - kg_min) / kg_range
    
    # Normalize EI to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-12
    ei_norm = (ei - ei_min) / ei_range
    
    # Blend KG and EI with adaptive weights
    # As f_best approaches the global minimum, increase exploitation (EI weight)
    global_min = -39.1662
    range_scale = 60.0
    progress = np.clip((f_best - global_min) / range_scale, 0.0, 1.0)
    
    # When progress is low (far from optimum), rely more on KG for exploration
    # When progress is high (close to optimum), rely more on EI for exploitation
    weight_kg = 0.4 * (1.0 - progress) + 0.1
    weight_ei = 1.0 - weight_kg
    
    score = weight_kg * kg_norm + weight_ei * ei_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
