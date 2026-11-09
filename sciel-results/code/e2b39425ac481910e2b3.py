"""Cost-aware acquisition for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
_DEADLINE_START = time.time()


def _value_density_cost(mu, sigma, f_best, cost):
    """
    Stochastic knapsack style value density:
    P(f < f_best) * E[improvement | f < f_best] / cost
    
    This is equivalent to EI / cost but computed in a numerically stable way
    that emphasizes the probabilistic efficiency of budget consumption.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-6)
    
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement: P(f < f_best) = Phi(z)
    prob_improve = norm.cdf(z)
    
    # Expected improvement magnitude given improvement occurs:
    # E[f_best - f | f < f_best] = (f_best - mu) * Phi(z) / Phi(z) + sigma * phi(z) / Phi(z)
    # Actually, EI = (f_best - mu)*Phi(z) + sigma*phi(z)
    # So E[improvement | improve] = EI / Phi(z) when Phi(z) > 0
    # But for the product P * E[improvement|improve] = EI, so it's the same.
    
    # Let's compute EI directly with numerical stability
    ei = np.zeros_like(mu)
    mask = z > -5.0  # For very negative z, EI is essentially 0
    if np.any(mask):
        z_m = z[mask]
        mu_m = mu[mask]
        sig_m = sigma_safe[mask]
        ei[mask] = (f_best - mu_m) * norm.cdf(z_m) + sig_m * norm.pdf(z_m)
    
    # Ensure non-negative
    ei = np.maximum(ei, 0.0)
    
    return ei / cost_safe


def _exploration_density_cost(mu, sigma, f_best, cost, beta=1.5):
    """
    Exploration-focused: score based on how much lower the LCB is than f_best,
    weighted by the uncertainty, divided by cost.
    
    This captures the "potential" of a point: if the lower tail of the GP
    posterior is well below f_best, and the cost is low, it's worth evaluating.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-6)
    
    # LCB: mu - beta * sigma
    lcb = mu - beta * sigma_safe
    
    # Improvement potential: how much better than f_best the LCB could be
    potential = np.maximum(f_best - lcb, 0.0)
    
    # Weight by uncertainty to avoid over-exploring already certain regions
    # Actually, the LCB already accounts for uncertainty via beta*sigma.
    # Let's just use the potential / cost
    return potential / cost_safe


def _thompson_sampling_score(mu, sigma, f_best, cost, rng=None):
    """
    Thompson sampling style: sample from posterior, rank by sampled value.
    Cost-aware: divide by cost.
    
    Since we need to return scores for 2000 points and pick the best,
    we can use a deterministic approximation: mu - alpha * sigma, where alpha
    is chosen to balance exploration and exploitation.
    
    For a more principled approach, we can use the quantile of the posterior
    that corresponds to the expected improvement threshold.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Use a quantile-based score: the 25th percentile of the posterior
    # This is more exploratory than the mean
    q25 = mu - 0.674 * sigma_safe  # ~25th percentile
    
    # Score: how much better than f_best the 25th percentile is
    score = np.maximum(f_best - q25, 0.0) / cost_safe
    
    return score


def _adaptive_value_density(mu, sigma, f_best, cost):
    """
    Adaptive combination of value density (exploitation) and exploration density.
    
    The key insight from the directive: model as stochastic knapsack, maximizing
    expected regret reduction per unit budget.
    
    Early in optimization: more exploration (high uncertainty regions)
    Late in optimization: more exploitation (regions likely to improve f_best)
    """
    # Ensure finite inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-10, posinf=1e6, neginf=1e-10)
    sigma = np.maximum(sigma, 1e-10)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6)
    cost = np.maximum(cost, 1e-6)
    
    # Check time budget
    elapsed = time.time() - _DEADLINE_START
    if elapsed > 310:  # Safety margin
        # Simple fallback: value density
        z = (f_best - mu) / sigma
        ei = np.maximum((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z), 0.0)
        return np.nan_to_num(ei / cost, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute value density (exploitation)
    vd_score = _value_density_cost(mu, sigma, f_best, cost)
    
    # Compute exploration density
    ed_score = _exploration_density_cost(mu, sigma, f_best, cost, beta=1.5)
    
    # Compute Thompson-style score
    ts_score = _thompson_sampling_score(mu, sigma, f_best, cost)
    
    # Normalize each score
    def normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mean_x = np.mean(x)
        std_x = np.std(x)
        if std_x < 1e-10:
            return x - mean_x
        return (x - mean_x) / std_x
    
    vd_norm = normalize(vd_score)
    ed_norm = normalize(ed_score)
    ts_norm = normalize(ts_score)
    
    # Adaptive weights based on progress
    # f_best close to -0.8 means we're near the optimum
    min_possible = -0.8
    # Upper bound: cosine8 can be up to 8 (sum of 8 cosines)
    # But in practice, with the shift, it's probably in [-0.8, 2]
    max_possible = 2.0
    progress = 1.0 - min(max((f_best - min_possible) / (max_possible - min_possible), 0.0), 1.0)
    
    # Early on (high progress): more exploration
    # Late (low progress): more exploitation
    w_vd = 0.40 + 0.30 * (1.0 - progress)  # Exploitation weight increases as we progress
    w_ed = 0.35 + 0.20 * progress  # Exploration weight decreases as we progress
    w_ts = 0.25  # Thompson sampling weight
    
    # Normalize weights
    total_w = w_vd + w_ed + w_ts
    w_vd /= total_w
    w_ed /= total_w
    w_ts /= total_w
    
    # Combined score
    score = w_vd * vd_norm + w_ed * ed_norm + w_ts * ts_norm
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Final time check
    if time.time() - _DEADLINE_START > 315:
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    array of shape (2000,) - acquisition scores (higher = better)
    """
    return _adaptive_value_density(mu, sigma, f_best, cost)
# EVOLVE-BLOCK-END
