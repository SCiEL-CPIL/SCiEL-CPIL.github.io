"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
TIME_LIMIT = 325.0  # Safety margin before 330s kill


def _time_exceeded():
    return (time.time() - _start_time) > TIME_LIMIT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Draws samples from the posterior N(mu, sigma) and selects candidates
    that minimize the sampled objective value per unit cost. This naturally
    balances exploration and exploitation through stochastic sampling.
    
    To handle the fact that we need to return scores for all 2000 candidates
    (not just select one), we use a multi-sample Thompson Sampling approach:
    draw multiple samples from the posterior and compute the expected
    minimum cost-normalized value across samples.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value (lower is better).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-10)
    
    # Check if we're running out of time
    if _time_exceeded():
        # Fallback to a simple EI-based score
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -10.0, 10.0)
        ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
        ei = np.maximum(ei, 0.0)
        score = ei / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return np.maximum(score, 0.0)
    
    # Thompson Sampling: draw multiple samples from the posterior
    # and compute the expected minimum of the cost-normalized objective
    n_samples = 50
    
    # Generate random normal values for all samples at once
    # Shape: (n_samples, 2000)
    rng = np.random.default_rng()
    Z = rng.standard_normal((n_samples, len(mu)))
    
    # Sample from posterior: y = mu + sigma * Z
    # Shape: (n_samples, 2000)
    samples = mu[None, :] + sigma_safe[None, :] * Z
    
    # For each sample, compute the cost-normalized objective
    # We want to find candidates that minimize the sampled value per unit cost
    # score_sample[i, j] = -samples[i, j] / cost_safe[j]
    # (negative because higher score = better, and we want to minimize the objective)
    scores_per_sample = -samples / cost_safe[None, :]
    
    # For Thompson Sampling, the acquisition score for candidate j is the
    # expected minimum of the cost-normalized objective across samples
    # Actually, a better approach: for each candidate, compute the probability
    # that it would be selected (i.e., have the minimum cost-normalized value)
    # weighted by how good that selection would be
    
    # Alternative: compute the expected minimum value across samples for each candidate
    # This gives us a sense of how "promising" each candidate is
    min_per_sample = np.min(scores_per_sample, axis=1)  # Shape: (n_samples,)
    
    # The expected minimum across samples
    expected_min = np.mean(min_per_sample)
    
    # For each candidate, compute how often it would be the argmin
    # and what the average value would be when it is the argmin
    is_argmin = (scores_per_sample == min_per_sample[:, None])  # Shape: (n_samples, 2000)
    
    # Count how many times each candidate is the argmin
    argmin_count = np.sum(is_argmin, axis=0)  # Shape: (2000,)
    
    # Compute the average score when each candidate is the argmin
    argmin_scores = np.where(is_argmin, scores_per_sample, 0.0)
    avg_argmin_score = np.sum(argmin_scores, axis=0) / np.maximum(argmin_count, 1)
    
    # Thompson Sampling score: probability of being selected * expected quality
    # Higher is better
    prob_selected = argmin_count / n_samples
    ts_score = prob_selected * avg_argmin_score
    
    # Also add a deterministic component based on EI for stability
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    ei_score = ei / cost_safe
    
    # Combine Thompson Sampling score with EI score
    # Normalize both to have similar scales
    ts_score_normalized = ts_score / (np.max(np.abs(ts_score)) + 1e-10)
    ei_score_normalized = ei_score / (np.max(ei_score) + 1e-10)
    
    # Weighted combination: emphasize Thompson Sampling for exploration
    combined = 0.7 * ts_score_normalized + 0.3 * ei_score_normalized
    
    # Ensure finite scores
    score = np.where(np.isfinite(combined), combined, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
