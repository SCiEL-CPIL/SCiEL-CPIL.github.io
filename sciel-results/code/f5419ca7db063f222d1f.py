# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _adaptive_lcb(mu, sigma, f_best, cost, beta_base=1.5, cost_exp=1.0, gamma=0.5):
    """Adaptive LCB with cost-awareness.
    
    The beta parameter scales with the relative uncertainty (sigma / (f_best - mu))
    to avoid premature convergence. Points with high uncertainty relative to their
    potential improvement get more exploration credit.
    
    score = -(mu - beta * sigma) / cost^cost_exp
    where beta = beta_base * (1 + gamma * relative_uncertainty)
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Relative uncertainty: how much sigma matters relative to potential improvement
    # If mu is far below f_best, even small sigma could be important
    # If mu is close to f_best, we need larger sigma to justify exploration
    potential = np.maximum(f_best - mu, 1e-9)
    rel_uncertainty = sigma_safe / potential
    
    # Adaptive beta: more exploration when relative uncertainty is high
    beta = beta_base * (1.0 + gamma * np.log1p(rel_uncertainty))
    
    # LCB score: we want to minimize mu - beta*sigma, so score = -(mu - beta*sigma)
    # Higher score = more promising
    lcb = mu - beta * sigma_safe
    score = -lcb / (cost_safe ** cost_exp)
    
    return score


def _ei_cost(mu, sigma, f_best, cost, cost_exp=1.0):
    """Standard EI with cost discounting."""
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    score = ei / (cost_safe ** cost_exp)
    
    return score


def _hybrid_score(mu, sigma, f_best, cost, w_lcb=0.6, w_ei=0.4, 
                  beta_base=1.5, cost_exp=1.0, gamma=0.5):
    """Hybrid of adaptive LCB and cost-aware EI."""
    s_lcb = _adaptive_lcb(mu, sigma, f_best, cost, beta_base, cost_exp, gamma)
    s_ei = _ei_cost(mu, sigma, f_best, cost, cost_exp)
    
    # Normalize both to [0, 1] for blending
    def _normalize(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-12:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    s_lcb_norm = _normalize(s_lcb)
    s_ei_norm = _normalize(s_ei)
    
    score = w_lcb * s_lcb_norm + w_ei * s_ei_norm
    return score


def _quick_calibrate(mu, sigma, f_best, cost, time_budget=3.0):
    """Quick calibration of hyperparameters by evaluating a proxy objective.
    
    We want the top-k scored points to have low mu (exploitation) while
    maintaining some diversity (not all from the same region). We use a
    simple proxy: minimize the mean mu of top-k points while maximizing
    the spread of their costs (to avoid clustering on cheap points only).
    """
    t0 = time.time()
    
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    top_k = min(10, len(mu) // 20)
    if top_k < 1:
        top_k = 1
    
    best_proxy = -np.inf
    best_params = (0.6, 0.4, 1.5, 1.0, 0.5)  # w_lcb, w_ei, beta_base, cost_exp, gamma
    
    # Small grid search
    w_lcb_vals = [0.5, 0.6, 0.7, 0.8]
    beta_vals = [1.0, 1.5, 2.0, 2.5]
    cost_exp_vals = [0.8, 1.0, 1.2]
    gamma_vals = [0.3, 0.5, 0.8]
    
    for w_lcb in w_lcb_vals:
        w_ei = 1.0 - w_lcb
        for beta_base in beta_vals:
            for cost_exp in cost_exp_vals:
                for gamma in gamma_vals:
                    if time.time() - t0 > time_budget:
                        break
                    
                    scores = _hybrid_score(mu_safe, sigma_safe, f_best, cost_safe,
                                           w_lcb, w_ei, beta_base, cost_exp, gamma)
                    
                    if not np.all(np.isfinite(scores)):
                        continue
                    
                    top_idx = np.argsort(scores)[-top_k:]
                    mean_mu = np.mean(mu_safe[top_idx])
                    cost_spread = np.std(cost_safe[top_idx]) + 1e-9
                    
                    # We want low mean_mu and reasonable cost_spread
                    proxy = -mean_mu + 0.1 * np.log(cost_spread)
                    
                    if proxy > best_proxy:
                        best_proxy = proxy
                        best_params = (w_lcb, w_ei, beta_base, cost_exp, gamma)
    
    return best_params


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for Hartmann 3D minimization."""
    t_start = time.time()
    deadline = t_start + 325.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Calibrate hyperparameters
    w_lcb, w_ei, beta_base, cost_exp, gamma = _quick_calibrate(
        mu_safe, sigma_safe, f_best, cost_safe, time_budget=3.0
    )
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: simple adaptive LCB
        scores = _adaptive_lcb(mu_safe, sigma_safe, f_best, cost_safe, 1.5, 1.0, 0.5)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Compute hybrid score
    scores = _hybrid_score(mu_safe, sigma_safe, f_best, cost_safe,
                           w_lcb, w_ei, beta_base, cost_exp, gamma)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
