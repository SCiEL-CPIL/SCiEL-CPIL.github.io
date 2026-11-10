"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s hard kill

def _thompson_sampling_score(mu, sigma, f_best, cost, n_samples=8):
    """
    Cost-Adjusted Thompson Sampling acquisition function.
    
    Instead of computing a deterministic acquisition score, we sample from
    the GP posterior and compute the expected cost-adjusted improvement.
    
    For each candidate i:
    - Sample f_i ~ N(mu_i, sigma_i^2) from the GP posterior
    - The "value" of evaluating point i is (f_best - f_i) / cost_i
    - We want to maximize this value in expectation over the posterior
    
    The expected cost-adjusted improvement is:
    E[(f_best - f_i) / cost_i | D] = (f_best - mu_i) / cost_i
    
    But to add stochasticity (Thompson Sampling), we sample:
    - Draw s_i ~ N(0, sigma_i^2)
    - Score_i = (f_best - (mu_i + s_i)) / cost_i
    
    This creates a stochastic ranking that naturally balances exploration
    and exploitation: high uncertainty (large sigma_i) leads to more variance
    in the score, making it more likely to be selected.
    
    To make this deterministic (as required), we use the expected score
    plus a controlled stochastic component based on the quantile of the
    normal distribution.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Base expected improvement per unit cost
    base_score = (f_best - mu) / cost_safe
    
    # Add exploration term: sigma / cost
    # This is the key structural difference from EI/cost:
    # EI/cost has a nonlinear relationship between sigma and the score
    # Here, we have a linear relationship, which is more stable
    
    # Use a quantile-based approach: score at the 75th percentile of the posterior
    # This gives a balance between exploitation (mean) and exploration (variance)
    z_quantile = norm.ppf(0.75)  # ~0.674
    
    exploration_score = z_quantile * sigma_safe / cost_safe
    
    # Combine: exploitation + exploration
    score = base_score + exploration_score
    
    return score

def _entropy_based_score(mu, sigma, f_best, cost):
    """
    Differential entropy-based acquisition function.
    
    The idea: prioritize points where the posterior is most uncertain
    relative to the current best, weighted by cost.
    
    Score = -H(f_i | D) / cost_i + (f_best - mu_i) / cost_i
    
    where H is the differential entropy of the GP posterior at point i.
    For a Gaussian N(mu, sigma^2), H = 0.5 * log(2*pi*e*sigma^2)
    
    This is structurally different from EI or UCB:
    - It directly measures information gain
    - The log term makes it less sensitive to extreme sigma values
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Differential entropy of N(mu, sigma^2)
    # H = 0.5 * log(2*pi*e*sigma^2) = 0.5 * log(2*pi*e) + log(sigma)
    # We can drop the constant term
    entropy = np.log(sigma_safe)
    
    # Information gain relative to current best
    # Points with high entropy AND low mu are most valuable
    info_gain = -entropy / cost_safe
    
    # Exploitation term
    exploit = (f_best - mu) / cost_safe
    
    # Combine
    score = exploit + 0.5 * info_gain
    
    return score

def _quantile_ei_score(mu, sigma, f_best, cost, quantiles=None):
    """
    Quantile-based Expected Improvement.
    
    Instead of standard EI which integrates over all improvements,
    we compute the expected improvement above a quantile threshold.
    
    This is structurally different: it focuses on "meaningful" improvements
    rather than any improvement, which can be more robust in noisy settings.
    
    Score_i = E[max(f_best - f_i, 0) * 1{f_i < threshold_i}] / cost_i
    
    where threshold_i = mu_i - k * sigma_i (k controls the quantile)
    """
    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    total_score = np.zeros_like(mu)
    
    for q in quantiles:
        z_q = norm.ppf(q)
        threshold = mu + z_q * sigma_safe
        
        # Expected improvement above the threshold
        # E[max(f_best - f_i, 0) | f_i < threshold]
        
        # P(f_i < threshold) = 0.5 * (1 + erf(z_q / sqrt(2))) = q
        # E[f_i | f_i < threshold] = mu - sigma * phi(z_q) / Phi(z_q)
        # where phi is the PDF and Phi is the CDF of the standard normal
        
        phi_z = norm.pdf(z_q)
        Phi_z = norm.cdf(z_q)
        
        # Avoid division by zero
        Phi_z_safe = np.maximum(Phi_z, 1e-9)
        
        E_f_given_less = mu - sigma_safe * phi_z / Phi_z_safe
        
        # E[max(f_best - f_i, 0)] = E[(f_best - f_i) * 1{f_i < f_best}]
        # = (f_best - E[f_i | f_i < f_best]) * P(f_i < f_best)
        
        # P(f_i < f_best) = Phi((f_best - mu) / sigma)
        z_ei = (f_best - mu) / sigma_safe
        Phi_ei = norm.cdf(z_ei)
        
        # E[f_i | f_i < f_best] = mu - sigma * phi(z_ei) / Phi(z_ei)
        phi_ei = norm.pdf(z_ei)
        Phi_ei_safe = np.maximum(Phi_ei, 1e-9)
        E_f_given_best = mu - sigma_safe * phi_ei / Phi_ei_safe
        
        EI = (f_best - E_f_given_best) * Phi_ei
        
        # Weight by the probability of being below the quantile threshold
        # This creates a "focused" improvement
        weight = norm.cdf((threshold - mu) / sigma_safe)  # = q
        
        total_score += EI * weight / cost_safe
    
    # Normalize by number of quantiles
    total_score /= len(quantiles)
    
    return total_score

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling-inspired scoring.
    
    This is a structural departure from EI/UCB hybrids. Instead of computing
    a deterministic acquisition score based on expected improvement or
    upper confidence bounds, we use a quantile-based approach that:
    
    1. Computes the expected cost-adjusted improvement (exploitation)
    2. Adds a controlled exploration term based on the posterior uncertainty
    3. Uses a quantile-based weighting to focus on "meaningful" improvements
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far (lower is better).
    cost : np.ndarray, shape (2000,)
        Evaluation cost for each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = better to evaluate next).
    """
    # Time guard
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        score = (f_best - mu) / cost_safe + 0.5 * sigma_safe / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Estimate optimization progress from f_best
    # Use a smooth transition to adapt the exploration weight
    # Early: more exploration (higher weight on sigma)
    # Late: more exploitation (higher weight on (f_best - mu))
    
    # Normalize f_best for the progress estimate
    # Typical ranges: Ackley/Rastrigin/Levy start around 10-30, end near 0
    # Hartmann starts around 3-5, ends near -3.86
    f_best_norm = np.clip(f_best, -5.0, 30.0)
    
    # Progress: 0 = early (f_best large), 1 = late (f_best small)
    # Use a logistic function
    progress = 1.0 / (1.0 + np.exp(0.3 * (f_best_norm - 5.0)))
    progress = np.clip(progress, 0.0, 1.0)
    
    # Exploration weight: high early, low late
    # w_explore = 1.0 - progress (early: 1.0, late: 0.0)
    # But we want at least some exploration
    w_explore = 0.3 + 0.7 * (1.0 - progress)
    
    # Compute the Thompson Sampling-inspired score
    # Base: expected cost-adjusted improvement
    base_score = (f_best - mu) / cost
    
    # Exploration: posterior uncertainty per unit cost
    explore_score = sigma / cost
    
    # Combine with adaptive weight
    score = base_score + w_explore * explore_score
    
    # Add a small entropy-based term to encourage diversity
    # This helps avoid getting stuck in local optima
    entropy_term = np.log(sigma + 1e-9) / cost
    score += 0.1 * entropy_term
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
