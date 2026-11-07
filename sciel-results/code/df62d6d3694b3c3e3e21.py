"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """Standard Expected Improvement."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _lower_confidence_bound(mu, sigma, f_best, kappa=2.576):
    """Lower Confidence Bound (UCB) - maximized for minimization."""
    return mu - kappa * sigma


def _probability_of_improvement(mu, sigma, f_best):
    """Probability of Improvement."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)


def _knowledge_gradient(mu, sigma, f_best, xi=0.0):
    """Approximate Knowledge Gradient (simplified)."""
    # Use EI as a proxy but with different weighting
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Weight by uncertainty to balance exploration
    uncertainty_weight = 1.0 + np.log1p(sigma_safe / max(abs(f_best), 1e-6))
    return np.maximum(ei, 0.0) * uncertainty_weight


def _acq_combination(mu, sigma, f_best, alpha=0.5, beta=0.3, gamma=0.2):
    """Combination of multiple acquisition functions."""
    ei = _expected_improvement(mu, sigma, f_best, xi=0.01)
    ucb = _lower_confidence_bound(mu, sigma, f_best, kappa=2.576)
    poi = _probability_of_improvement(mu, sigma, f_best)
    
    # Normalize each to [0, 1] range for combination
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_min = np.min(ucb)
    ucb_range = np.max(ucb) - ucb_min if np.max(ucb) - ucb_min > 0 else 1.0
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Combine
    score = alpha * (ei / ei_max) + beta * ucb_norm + gamma * poi
    return score


def _adaptive_acquisition(mu, sigma, f_best, step_count=0):
    """Adaptive acquisition that changes strategy based on progress."""
    n = len(mu)
    
    # Early stages: more exploration (UCB-like)
    # Middle stages: balanced (EI)
    # Late stages: more exploitation (POI-like)
    
    if step_count < 10:
        # Exploration phase
        return _lower_confidence_bound(mu, sigma, f_best, kappa=3.0)
    elif step_count < 20:
        # Transition phase
        return _acq_combination(mu, sigma, f_best, alpha=0.5, beta=0.3, gamma=0.2)
    else:
        # Exploitation phase
        return _expected_improvement(mu, sigma, f_best, xi=0.005)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    
    Uses an adaptive strategy combining multiple acquisition functions.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    n = len(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Try different acquisition strategies and pick the best
    # Strategy 1: Standard EI
    scores_ei = _expected_improvement(mu, sigma, f_best, xi=0.01)
    
    # Strategy 2: UCB
    scores_ucb = _lower_confidence_bound(mu, sigma, f_best, kappa=2.576)
    
    # Strategy 3: Combination
    scores_comb = _acq_combination(mu, sigma, f_best, alpha=0.4, beta=0.35, gamma=0.25)
    
    # Strategy 4: Knowledge Gradient approximation
    scores_kg = _knowledge_gradient(mu, sigma, f_best, xi=0.01)
    
    # Check time
    if time.time() > deadline:
        # Return the best available result
        return scores_ei
    
    # Use a weighted combination of the top strategies
    # Normalize each score set to [0, 1]
    def normalize(scores):
        s_min = np.min(scores)
        s_max = np.max(scores)
        if s_max - s_min < 1e-12:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)
    
    n_ei = normalize(scores_ei)
    n_ucb = normalize(scores_ucb)
    n_comb = normalize(scores_comb)
    n_kg = normalize(scores_kg)
    
    # Final combination
    final_scores = 0.35 * n_ei + 0.25 * n_ucb + 0.25 * n_comb + 0.15 * n_kg
    
    # Ensure finite values
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return final_scores
# EVOLVE-BLOCK-END
