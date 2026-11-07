# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import ndtri
from scipy.optimize import minimize_scalar
from scipy.stats import norm

_t0 = None
_DL = None

def _elapsed():
    if _t0 is None:
        return 0.0
    return time.time() - _t0

def _norm_cdf_arr(z):
    return 0.5 * (1.0 + np.tanh(0.7071067811865476 * z))

def _norm_pdf_arr(z):
    return np.exp(-0.5 * z * z) * 0.3989422804014327

def _ei_min_vec(mu, sigma, f_best):
    eps = 1e-12
    s = np.maximum(sigma, eps)
    z = (f_best - mu) / s
    cdf = _norm_cdf_arr(z)
    pdf = _norm_pdf_arr(z)
    ei = (f_best - mu) * cdf + s * pdf
    return np.maximum(ei, 0.0)

def _qoei_min(mu, sigma, f_best, q):
    eps = 1e-12
    s = np.maximum(sigma, eps)
    z = (f_best - mu) / s
    z_q = ndtri(q)
    d = z - z_q
    cdf_z = _norm_cdf_arr(z)
    pdf_z = _norm_pdf_arr(z)
    # E[(y - f_best)*1{y < f_best}] = (f_best - mu)*Phi(z) + s*phi(z)
    # E[(y - f_best)*1{y >= f_best - q*s}]
    # = (f_best - mu) * P(y >= f_best - q*s) + s * E[(y-mu)/s * 1{y >= f_best-q*s}]
    # P(y >= f_best - q*s) = P((y-mu)/s >= f_best - q*s - mu)/s) = 1 - Phi((f_best-mu)/s - q) = 1 - Phi(z - q)
    # E[(y-mu)/s * 1{...}] = phi(z-q) - (z-q)*(1-Phi(z-q))
    d2 = z - q
    cdf_d2 = _norm_cdf_arr(d2)
    pdf_d2 = _norm_pdf_arr(d2)
    term1 = (f_best - mu) * (1.0 - cdf_d2)
    term2 = s * (pdf_d2 - d2 * (1.0 - cdf_d2))
    ei_full = (f_best - mu) * cdf_z + s * pdf_z
    qoei = term1 + term2 - ei_full
    # qoei should be non-negative when q is between 0 and 1
    # But due to numerical issues, clip
    return np.maximum(qoei, 0.0)

def _ucb_vec(mu, sigma, beta):
    return -mu + beta * np.maximum(sigma, 1e-12)

def _minmax(a):
    a = np.asarray(a, dtype=float)
    lo = np.nanmin(a)
    hi = np.nanmax(a)
    rng = hi - lo
    if rng < 1e-12:
        return np.ones_like(a) * 0.5
    out = (a - lo) / rng
    return np.nan_to_num(out, nan=0.5, posinf=0.5, neginf=0.5)

def _score_components(mu, sigma, f_best, cost, q, beta):
    """Compute all components. Returns dict of normalized scores."""
    n = len(mu)
    if n < 2:
        return None
    # EI
    ei = _ei_min_vec(mu, sigma, f_best)
    # QOEI
    qoei = _qoei_min(mu, sigma, f_best, q)
    # UCB
    ucb = _ucb_vec(mu, sigma, beta)
    # Cost efficiency: prefer low cost
    cost_eff = 1.0 / (cost + 1e-12)
    # Cost-adjusted EI: EI / cost
    ei_cost = ei / (cost + 1e-12)
    # Cost-adjusted QOEI
    qoei_cost = qoei / (cost + 1e-12)
    # Cost-adjusted UCB
    ucb_cost = ucb / (cost + 1e-12)
    
    return {
        'ei': _minmax(ei),
        'qoei': _minmax(qoei),
        'ucb': _minmax(ucb),
        'cost_eff': _minmax(cost_eff),
        'ei_cost': _minmax(ei_cost),
        'qoei_cost': _minmax(qoei_cost),
        'ucb_cost': _minmax(ucb_cost),
    }

def _greedy_budget(mu, sigma, f_best, cost, score_vec, budget):
    """
    Simulate a budgeted greedy search.
    At each step, pick the candidate with highest score among those with cost <= remaining.
    Track the best value found (using mu as a proxy for the true value).
    """
    n = len(mu)
    remaining = budget
    visited = np.zeros(n, dtype=bool)
    best_val = f_best
    
    for _ in range(min(15, n)):
        feasible = (~visited) & (cost <= remaining + 1e-9)
        if not np.any(feasible):
            break
        cand_score = np.where(feasible, score_vec, -np.inf)
        pick = int(np.argmax(cand_score))
        if not feasible[pick]:
            break
        visited[pick] = True
        remaining -= cost[pick]
        if mu[pick] < best_val:
            best_val = mu[pick]
    
    return best_val

def _fit_acquisition(mu, sigma, f_best, cost):
    """
    Fit acquisition parameters by simulating budgeted greedy search.
    We search over:
    - q: quantile for QOEI (0 to 1, lower = more aggressive)
    - beta: UCB exploration
    - weights: mix of EI, QOEI, UCB, cost_eff, and their cost-adjusted versions
    """
    n = len(mu)
    if n < 2:
        return 0.5, 1.0, (0.3, 0.2, 0.2, 0.1, 0.2, 0.0, 0.0)
    
    # Budget estimate: total 30, initial 5 uniform points cost ~5*1.5=7.5
    # Remaining ~22.5. But cost varies. Let's use 20 as a safe estimate.
    budget = 20.0
    
    # Precompute all score components for each (q, beta) combination
    # We'll use a small grid
    q_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    beta_grid = [0.5, 1.0, 1.5, 2.0, 3.0]
    
    # Weight grid: we'll use a few combinations
    # w_ei, w_qoei, w_ucb, w_cost_eff, w_ei_cost, w_qoei_cost, w_ucb_cost
    # Sum to 1
    w_grid = [
        (0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0),
        (0.2, 0.4, 0.2, 0.1, 0.1, 0.0, 0.0),
        (0.1, 0.3, 0.2, 0.1, 0.2, 0.1, 0.0),
        (0.0, 0.4, 0.2, 0.1, 0.1, 0.2, 0.0),
        (0.2, 0.3, 0.2, 0.1, 0.0, 0.1, 0.1),
        (0.1, 0.3, 0.3, 0.1, 0.0, 0.1, 0.1),
        (0.0, 0.4, 0.3, 0.1, 0.0, 0.1, 0.1),
        (0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.0),
        (0.2, 0.3, 0.2, 0.1, 0.1, 0.1, 0.0),
        (0.1, 0.4, 0.2, 0.1, 0.1, 0.1, 0.0),
    ]
    
    best_regret = np.inf
    best_q = 0.5
    best_beta = 1.0
    best_w = (0.3, 0.2, 0.2, 0.1, 0.2, 0.0, 0.0)
    
    # Precompute score components for each (q, beta)
    score_cache = {}
    for q in q_grid:
        for beta in beta_grid:
            key = (q, beta)
            comps = _score_components(mu, sigma, f_best, cost, q, beta)
            if comps is not None:
                score_cache[key] = comps
    
    for (w_ei, w_qoei, w_ucb, w_ce, w_eic, w_qoc, w_ucbc) in w_grid:
        for q in q_grid:
            for beta in beta_grid:
                key = (q, beta)
                if key not in score_cache:
                    continue
                comps = score_cache[key]
                
                score_vec = (
                    w_ei * comps['ei'] +
                    w_qoei * comps['qoei'] +
                    w_ucb * comps['ucb'] +
                    w_ce * comps['cost_eff'] +
                    w_eic * comps['ei_cost'] +
                    w_qoc * comps['qoei_cost'] +
                    w_ucbc * comps['ucb_cost']
                )
                score_vec = np.nan_to_num(score_vec, nan=0.0, posinf=0.0, neginf=0.0)
                
                best_val = _greedy_budget(mu, sigma, f_best, cost, score_vec, budget)
                regret = f_best - best_val  # Lower is better (more improvement)
                
                if regret < best_regret:
                    best_regret = regret
                    best_q = q
                    best_beta = beta
                    best_w = (w_ei, w_qoei, w_ucb, w_ce, w_eic, w_qoc, w_ucbc)
    
    return best_q, best_beta, best_w

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on hartmann 6d.
    Uses QOEI, EI, UCB, and cost-aware components, with parameters fitted
    via budgeted greedy simulation.
    """
    global _t0, _DL
    if _t0 is None:
        _t0 = time.time()
        _DL = _t0 + 320.0
    
    if time.time() > _DL:
        # Emergency fallback
        ei = _ei_min_vec(mu, sigma, f_best)
        ce = 1.0 / (cost + 1e-12)
        out = 0.8 * _minmax(ei) + 0.2 * _minmax(ce)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    n = len(mu)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    
    try:
        q, beta, w = _fit_acquisition(mu, sigma, f_best, cost)
    except Exception:
        q, beta, w = 0.5, 1.0, (0.3, 0.2, 0.2, 0.1, 0.2, 0.0, 0.0)
    
    # Build final score
    comps = _score_components(mu, sigma, f_best, cost, q, beta)
    if comps is None:
        # Fallback
        ei = _ei_min_vec(mu, sigma, f_best)
        return _minmax(ei)
    
    score = (
        w[0] * comps['ei'] +
        w[1] * comps['qoei'] +
        w[2] * comps['ucb'] +
        w[3] * comps['cost_eff'] +
        w[4] * comps['ei_cost'] +
        w[5] * comps['qoei_cost'] +
        w[6] * comps['ucb_cost']
    )
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize to [0, 1]
    mx = np.max(score)
    if mx > 1e-12:
        score = score / mx
    else:
        score = np.ones_like(score) * 0.5
    
    return score
# EVOLVE-BLOCK-END
