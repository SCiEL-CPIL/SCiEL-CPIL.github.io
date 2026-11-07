import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 300.0
_MARGIN = 15.0

def _remaining_time():
    return _TIME_LIMIT - (time.time() - _start_time) - _MARGIN

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running out of time, return a simple score
    if _remaining_time() < 0.5:
        cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        z = (f_best - mu_safe) / sigma_safe
        z = np.clip(z, -10, 10)
        ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure inputs are clean
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    f_best = float(np.where(np.isfinite(f_best), f_best, 0.0))
    
    # Compute z for EI
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Probability of Improvement (for exploration balance)
    pi = cdf_z
    
    # Hypervolume-inspired improvement:
    # The idea is that we want to maximize the expected reduction in the
    # "non-dominated" region. For a single-objective minimization, the
    # hypervolume of the non-dominated set relative to a reference point
    # is simply (f_best - ref) if f_best < ref, else 0.
    # 
    # A more principled approach: use the expected hypervolume improvement
    # which for single objective reduces to EI, but we can add a term that
    # accounts for the "area" under the GP that is below f_best.
    #
    # Alternative structural change: use a combination of EI and a
    # "regret bound" based term that encourages exploration in high-uncertainty
    # regions while being cost-aware.
    #
    # Let's try: Score = EI / cost + beta * sigma / cost * (1 - pi)
    # The second term rewards high uncertainty in regions where improvement
    # is likely (high pi), but scaled by (1-pi) to balance.
    #
    # Actually, let's try a different structural approach:
    # Use the "Knowledge Gradient" approximation or a "Expected Regret" based
    # acquisition that is more robust.
    #
    # For a structural change, let's use a "Thompson Sampling" inspired
    # approach: sample from the posterior and pick the candidate with the
    # lowest sampled value, but in a vectorized way.
    #
    # However, Thompson Sampling requires sampling, which is expensive.
    # Instead, let's use a "UCB" variant that is cost-aware and includes
    # an exploration term based on the GP variance.
    #
    # UCB: mu - kappa * sigma
    # Cost-aware: (mu - kappa * sigma) / cost
    # But we want to MINIMIZE, so higher score = better.
    # Score = -(mu - kappa * sigma) / cost = (kappa * sigma - mu) / cost
    #
    # Let's combine EI and UCB for a more robust acquisition:
    # Score = (EI + alpha * UCB_term) / cost
    # where UCB_term = -mu + beta * sigma (since we minimize, lower mu + higher sigma = better)
    
    # UCB term for minimization: we want low mu and high sigma
    # ucb_value = mu - gamma * sigma (lower is better)
    # score_ucb = -ucb_value / cost = (gamma * sigma - mu) / cost
    
    gamma = 2.0  # Exploration-exploitation balance
    ucb_score = (gamma * sigma_safe - mu) / cost
    
    # Normalize EI and UCB to combine them
    # EI is already non-negative for points with mu < f_best
    # Let's use a weighted combination
    alpha = 0.5  # Weight for EI vs UCB
    
    # Scale EI to be comparable to UCB
    # EI has units of objective, UCB also has units of objective
    score = (alpha * ei + (1 - alpha) * ucb_score) / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
