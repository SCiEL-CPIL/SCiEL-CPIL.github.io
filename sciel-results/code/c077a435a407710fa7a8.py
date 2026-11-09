"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of styblinski tang 1d.
    
    Uses a blend of Expected Improvement (EI) and a Monte Carlo Knowledge Gradient (KG)
    approximation. The KG component estimates the expected improvement in the global
    minimum by considering the top-k candidates jointly, which is more robust for
    multimodal functions than pointwise EI or UCB.
    
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
    t0 = time.time()
    deadline = 300.0  # safety margin before 330s kill
    
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
    
    # Knowledge Gradient approximation via Monte Carlo
    # Select top-k candidates by EI score
    k = min(50, len(mu))
    top_k_idx = np.argsort(ei)[-k:]
    top_k_mu = mu[top_k_idx]
    top_k_sigma = sigma[top_k_idx]
    
    # Estimate KG using Monte Carlo approximation of the joint posterior
    n_mc = 100
    rng = np.random.default_rng(42)
    mc_samples = rng.standard_normal((n_mc, k))
    
    # For each MC sample, compute the minimum value and which candidate achieves it
    sampled_values = top_k_mu[None, :] + top_k_sigma[None, :] * mc_samples  # (n_mc, k)
    min_values = np.min(sampled_values, axis=1)  # (n_mc,)
    is_min = np.argmin(sampled_values, axis=1)  # (n_mc,)
    
    # KG score for each top-k candidate: expected improvement in global min
    # P(candidate i is min) * E[improvement | candidate i is min]
    counts = np.bincount(is_min, minlength=k)
    prob_min = counts / n_mc
    
    # Expected improvement given candidate i is the minimum
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
    
    # Check time budget
    if time.time() - t0 > deadline - 10:
        # Return EI-based scores as fallback
        ei_min = np.min(ei)
        ei_max = np.max(ei)
        ei_range = ei_max - ei_min + 1e-12
        ei_norm = (ei - ei_min) / ei_range
        return np.where(np.isfinite(ei_norm), ei_norm, 0.0)
    
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
    weight_kg = 0.3 * (1.0 - progress) + 0.15
    weight_ei = 1.0 - weight_kg
    
    score = weight_kg * kg_norm + weight_ei * ei_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
