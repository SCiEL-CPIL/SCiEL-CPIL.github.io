"""Cost-aware acquisition for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
_DEADLINE_START = time.time()


def _expected_improvement_cost(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement for minimization.
    
    For minimization, improvement is (f_best - f(x)).
    EI = E[max(f_best - f(x), 0)]
    
    We divide by cost to get expected improvement per unit cost.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    # For minimization: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI formula for minimization
    # E[max(f_best - f, 0)] where f ~ N(mu, sigma^2)
    # = (f_best - mu) * Phi(z) + sigma * phi(z)  when z > 0
    # = 0 otherwise (approximately)
    ei = np.zeros_like(mu)
    mask = z > -0.5  # Only compute where there's meaningful improvement
    if np.any(mask):
        z_m = z[mask]
        mu_m = mu[mask]
        sig_m = sigma_safe[mask]
        ei[mask] = (f_best - mu_m) * norm.cdf(z_m) + sig_m * norm.pdf(z_m)
    
    # Cost-adjusted: divide by cost
    cost_safe = np.maximum(cost, 1e-6)
    return ei / cost_safe


def _lower_confidence_bound_cost(mu, sigma, f_best, cost, xi=0.0, kappa=1.0):
    """
    Cost-aware Lower Confidence Bound (for minimization).
    
    LCB = mu - kappa * sigma  (lower is better for minimization)
    We want to maximize negative LCB per unit cost.
    """
    lcb = mu - kappa * sigma
    cost_safe = np.maximum(cost, 1e-6)
    # Higher score = better: use -lcb / cost
    return -lcb / cost_safe


def _probability_of_improvement_cost(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement for minimization.
    
    PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    cost_safe = np.maximum(cost, 1e-6)
    return pi / cost_safe


def _thompson_sampling_cost(mu, sigma, f_best, cost, rng=None):
    """
    Cost-aware Thompson Sampling.
    
    Sample from posterior and pick the one with best value/cost ratio.
    For batch scoring, we use a deterministic approximation based on
    the posterior mean and uncertainty.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Sample from the GP posterior
    samples = mu + sigma * rng.standard_normal(len(mu))
    cost_safe = np.maximum(cost, 1e-6)
    
    # For minimization, we want low values. Score = -sample / cost
    score = -samples / cost_safe
    return score


def _knowledge_gradient_cost(mu, sigma, f_best, cost):
    """
    Approximate Knowledge Gradient: expected reduction in uncertainty
    about the minimum, weighted by cost.
    
    Simplified: use a combination of where the GP is uncertain (high sigma)
    and where it might improve the best.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Expected improvement in the best value
    z = (f_best - mu) / sigma_safe
    ei = np.zeros_like(mu)
    mask = z > -0.5
    if np.any(mask):
        z_m = z[mask]
        mu_m = mu[mask]
        sig_m = sigma_safe[mask]
        ei[mask] = (f_best - mu_m) * norm.cdf(z_m) + sig_m * norm.pdf(z_m)
    
    # Uncertainty bonus: high sigma where mu is close to f_best
    uncertainty_bonus = sigma_safe * np.exp(-((mu - f_best) ** 2) / (2 * (sigma_safe + 1e-10) ** 2))
    
    # Combine
    score = (ei + 0.5 * uncertainty_bonus) / cost_safe
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
    # Safety: ensure inputs are finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-10, posinf=1e6, neginf=1e-10)
    sigma = np.maximum(sigma, 1e-10)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6)
    cost = np.maximum(cost, 1e-6)
    
    # Check time budget
    elapsed = time.time() - _DEADLINE_START
    if elapsed > 315:  # Safety margin
        # Simple fallback
        return np.nan_to_num(-mu / cost, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute multiple acquisition functions
    # 1. Cost-aware Expected Improvement
    ei_score = _expected_improvement_cost(mu, sigma, f_best, cost)
    
    # 2. Cost-aware Lower Confidence Bound
    lcb_score = _lower_confidence_bound_cost(mu, sigma, f_best, cost, xi=0.0, kappa=1.0)
    
    # 3. Cost-aware Probability of Improvement
    pi_score = _probability_of_improvement_cost(mu, sigma, f_best, cost)
    
    # 4. Knowledge Gradient approximation
    kg_score = _knowledge_gradient_cost(mu, sigma, f_best, cost)
    
    # Normalize each score to have zero mean and unit variance
    def normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        std = np.std(x)
        if std < 1e-10:
            return x
        return (x - np.mean(x)) / std
    
    ei_norm = normalize(ei_score)
    lcb_norm = normalize(lcb_score)
    pi_norm = normalize(pi_score)
    kg_norm = normalize(kg_score)
    
    # Adaptive weights based on optimization progress
    # f_best close to -0.8 means we're near the optimum
    # f_best far from -0.8 means we need more exploration
    min_possible = -0.8
    max_possible = 0.8  # rough upper bound for cosine8
    progress = 1.0 - min(max((f_best - min_possible) / (max_possible - min_possible), 0.0), 1.0)
    
    # Early on (high progress): more exploration (LCB, KG)
    # Later (low progress): more exploitation (EI, PI)
    w_ei = 0.35 + 0.15 * (1.0 - progress)
    w_lcb = 0.25 + 0.10 * progress
    w_pi = 0.20 + 0.05 * (1.0 - progress)
    w_kg = 0.20 + 0.05 * progress
    
    # Ensure weights sum to 1
    total_w = w_ei + w_lcb + w_pi + w_kg
    w_ei /= total_w
    w_lcb /= total_w
    w_pi /= total_w
    w_kg /= total_w
    
    # Combined score
    score = w_ei * ei_norm + w_lcb * lcb_norm + w_pi * pi_norm + w_kg * kg_norm
    
    # Add a small bonus for low-cost points to encourage efficient exploration
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = max(cost_max - cost_min, 1e-10)
    cost_bonus = 0.05 * (1.0 - (cost - cost_min) / cost_range)
    
    score = score + cost_bonus
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Final time check
    if time.time() - _DEADLINE_START > 320:
        # Emergency fallback
        return np.nan_to_num(-mu / cost, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
