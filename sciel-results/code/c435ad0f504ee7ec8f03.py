"""Cost-aware acquisition for Hartmann-3D BO loop.
Combines Quantile-based LCB (structural) with calibrated hybrid weighting (refinement).
Uses a dynamic quantile that adapts to the cost landscape, and a sigmoidal
cost discount to aggressively prune expensive candidates in outer regions.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 325.0  # 330s limit, 5s safety margin


def _quantile_lcb(mu, sigma, f_best, q=0.05):
    """
    Quantile-based Lower Confidence Bound for minimization.
    Returns the q-th percentile of the predictive distribution, negated
    so that higher scores correspond to more promising (lower) values.
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    # q-th percentile of N(mu, sigma^2)
    z_q = norm.ppf(q)  # negative for q < 0.5
    lcb = mu + z_q * sigma_safe
    # Negate for higher-is-better
    return -lcb


def _sigmoid_cost_discount(cost, center=2.5, steepness=1.5):
    """
    Sigmoidal cost discount that aggressively penalizes high-cost candidates.
    Returns a value in (0, 1] where 1 means no penalty and lower means more penalty.
    Cost is in [1, 4] for x in [0,1]^3.
    """
    # Normalize cost to roughly [0, 1]
    cost_norm = (cost - 1.0) / 3.0  # [0, 1]
    # Sigmoid that is close to 1 for low cost and drops for high cost
    # center=0.5 means cost=2.5 gets 0.5 discount
    discount = 1.0 / (1.0 + np.exp(steepness * (cost_norm - 0.5)))
    # Clip to avoid extreme values
    return np.clip(discount, 0.1, 1.0)


def _expected_improvement(mu, sigma, f_best):
    """Standard Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    return ei


def _ucb_score(mu, sigma, f_best, kappa=2.0):
    """UCB acquisition for minimization: -mu + kappa*sigma."""
    return -mu + kappa * sigma


def _hybrid_acquisition(mu, sigma, f_best, cost, w_q=0.4, w_ei=0.3, w_ucb=0.3,
                        q=0.05, kappa=2.0, sigmoid_center=2.5, sigmoid_steep=1.5):
    """
    Hybrid acquisition combining Quantile-LCB, EI, and UCB with cost discounting.
    """
    # Compute each component
    q_score = _quantile_lcb(mu, sigma, f_best, q=q)
    ei_score = _expected_improvement(mu, sigma, f_best)
    ucb_score = _ucb_score(mu, sigma, f_best, kappa=kappa)
    
    # Normalize each to zero mean, unit std for fair combination
    def _normalize(x):
        x = np.asarray(x, dtype=np.float64)
        m = np.mean(x)
        s = np.std(x) + 1e-9
        return (x - m) / s
    
    q_norm = _normalize(q_score)
    ei_norm = _normalize(ei_score)
    ucb_norm = _normalize(ucb_score)
    
    # Weighted combination
    combined = w_q * q_norm + w_ei * ei_norm + w_ucb * ucb_norm
    
    # Apply sigmoidal cost discount
    discount = _sigmoid_cost_discount(cost, center=sigmoid_center, steepness=sigmoid_steep)
    score = combined * discount
    
    return score


def _calibrate(mu, sigma, f_best, cost, time_budget=2.0):
    """
    Lightweight calibration: try a small grid of parameters and pick the one
    that best ranks low-mu points highly among the top-k scored candidates.
    """
    t0 = time.time()
    
    # Parameter grid
    q_vals = [0.01, 0.05, 0.10]
    kappa_vals = [1.0, 2.0, 3.0]
    w_q_vals = [0.3, 0.5, 0.7]
    sigmoid_steep_vals = [1.0, 1.5, 2.0]
    
    best_proxy = -np.inf
    best_params = {'q': 0.05, 'kappa': 2.0, 'w_q': 0.5, 'w_ei': 0.25, 'w_ucb': 0.25,
                   'sigmoid_center': 2.5, 'sigmoid_steep': 1.5}
    
    top_k = min(10, len(mu) // 20)
    if top_k < 1:
        top_k = 1
    
    for q in q_vals:
        for kappa in kappa_vals:
            for w_q in w_q_vals:
                for steep in sigmoid_steep_vals:
                    if time.time() - t0 > time_budget:
                        break
                    
                    w_ei = (1.0 - w_q) * 0.5
                    w_ucb = (1.0 - w_q) * 0.5
                    
                    scores = _hybrid_acquisition(mu, sigma, f_best, cost,
                                                 w_q=w_q, w_ei=w_ei, w_ucb=w_ucb,
                                                 q=q, kappa=kappa,
                                                 sigmoid_center=2.5, sigmoid_steep=steep)
                    
                    if not np.all(np.isfinite(scores)):
                        continue
                    
                    top_idx = np.argsort(scores)[-top_k:]
                    mean_mu_top = np.mean(mu[top_idx])
                    proxy = -mean_mu_top
                    
                    if proxy > best_proxy:
                        best_proxy = proxy
                        best_params = {'q': q, 'kappa': kappa, 'w_q': w_q,
                                       'w_ei': w_ei, 'w_ucb': w_ucb,
                                       'sigmoid_center': 2.5, 'sigmoid_steep': steep}
    
    return best_params


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Safe fallback: simple quantile LCB
        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
        mu_safe = np.asarray(mu, dtype=np.float64)
        z_q = norm.ppf(0.05)
        lcb = mu_safe + z_q * sigma_safe
        scores = -lcb
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure inputs are proper arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Safety: handle non-finite values
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Calibrate parameters (quick, ~2s max)
    remaining = _TIME_LIMIT - (time.time() - _START_TIME)
    if remaining > 5.0:
        params = _calibrate(mu_safe, sigma_safe, f_best, cost_safe, time_budget=2.0)
    else:
        params = {'q': 0.05, 'kappa': 2.0, 'w_q': 0.5, 'w_ei': 0.25, 'w_ucb': 0.25,
                  'sigmoid_center': 2.5, 'sigmoid_steep': 1.5}
    
    # Compute hybrid score
    scores = _hybrid_acquisition(mu_safe, sigma_safe, f_best, cost_safe,
                                 w_q=params['w_q'], w_ei=params['w_ei'], w_ucb=params['w_ucb'],
                                 q=params['q'], kappa=params['kappa'],
                                 sigmoid_center=params['sigmoid_center'],
                                 sigmoid_steep=params['sigmoid_steep'])
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
