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
    d2 = z - q
    cdf_d2 = _norm_cdf_arr(d2)
    pdf_d2 = _norm_pdf_arr(d2)
    term1 = (f_best - mu) * (1.0 - cdf_d2)
    term2 = s * (pdf_d2 - d2 * (1.0 - cdf_d2))
    ei_full = (f_best - mu) * cdf_z + s * pdf_z
    qoei = term1 + term2 - ei_full
    return np.maximum(qoei, 0.0)

def _ucb_vec(mu, sigma, beta):
    return -mu + beta * np.maximum(sigma, 1e-12)

def _thompson_sample(mu, sigma, seed=None):
    """
    Draw a single sample from the posterior: mu + sigma * N(0,1).
    Return the sampled values (lower = better for minimization).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    z = rng.standard_normal(len(mu))
    return mu + np.maximum(sigma, 1e-12) * z

def _ts_score(mu, sigma, f_best, cost, n_samples=50, seed=42):
    """
    Thompson Sampling acquisition: draw multiple samples from the posterior,
    for each sample find the argmin (weighted by cost), and score candidates
    by how often they are selected and how good their sampled values are.
    
    This naturally balances exploration (high sigma -> more variance in samples)
    and exploitation (low mu -> more likely to be argmin).
    """
    n = len(mu)
    if n < 2:
        return np.ones(n) * 0.5
    
    # For each sample, we want to find the candidate that minimizes (sample_value / cost)
    # Score a candidate by: frequency of being selected * average quality when selected
    
    # Use a fixed seed for reproducibility within a call, but different seeds across calls
    rng = np.random.default_rng(seed)
    
    # Efficient approach: for each sample, compute value/cost and find top candidates
    # We'll track:
    # - count: how many times each candidate was in the top-k
    # - sum_value: sum of sampled values when selected
    
    count = np.zeros(n)
    sum_val = np.zeros(n)
    k = max(1, n // 100)  # top 1% of candidates per sample
    
    for _ in range(n_samples):
        z = rng.standard_normal(n)
        sample = mu + np.maximum(sigma, 1e-12) * z
        # Score for selection: lower sample/cost is better
        # We want to select candidates with low sample value and low cost
        # Use: sample / (cost + eps) as the criterion
        criterion = sample / (cost + 1e-12)
        
        # Find top-k by lowest criterion
        if k >= n:
            idx = np.arange(n)
        else:
            idx = np.argpartition(criterion, k)[:k]
        
        # Weight by how good the criterion is relative to the minimum
        min_c = np.min(criterion)
        # For selected candidates, track their sample value
        for i in idx:
            count[i] += 1
            sum_val[i] += sample[i]
    
    # Score: high count and low average sample value when selected
    # Normalize count
    count_norm = _minmax(count)
    # For sum_val, lower is better, so negate
    # But only for candidates that were selected at least once
    avg_val = np.where(count > 0, sum_val / np.maximum(count, 1), f_best)
    avg_val_norm = _minmax(-avg_val)  # negate so higher = better
    
    # Combine: 60% frequency, 40% quality
    score = 0.6 * count_norm + 0.4 * avg_val_norm
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

def _minmax(a):
    a = np.asarray(a, dtype=float)
    lo = np.nanmin(a)
    hi = np.nanmax(a)
    rng = hi - lo
    if rng < 1e-12:
        return np.ones_like(a) * 0.5
    out = (a - lo) / rng
    return np.nan_to_num(out, nan=0.5, posinf=0.5, neginf=0.5)

def _score_components(mu, sigma, f_best, cost, q, beta, ts_seed=None):
    """Compute all components. Returns dict of normalized scores."""
    n = len(mu)
    if n < 2:
        return None
    ei = _ei_min_vec(mu, sigma, f_best)
    qoei = _qoei_min(mu, sigma, f_best, q)
    ucb = _ucb_vec(mu, sigma, beta)
    cost_eff = 1.0 / (cost + 1e-12)
    ei_cost = ei / (cost + 1e-12)
    qoei_cost = qoei / (cost + 1e-12)
    ucb_cost = ucb / (cost + 1e-12)
    
    # Thompson Sampling score
    if ts_seed is None:
        ts_seed = int(np.sum(mu * 1000) % 10000)  # deterministic seed from data
    ts = _ts_score(mu, sigma, f_best, cost, n_samples=30, seed=ts_seed)
    
    return {
        'ei': _minmax(ei),
        'qoei': _minmax(qoei),
        'ucb': _minmax(ucb),
        'cost_eff': _minmax(cost_eff),
        'ei_cost': _minmax(ei_cost),
        'qoei_cost': _minmax(qoei_cost),
        'ucb_cost': _minmax(ucb_cost),
        'ts': ts,
    }

def _greedy_budget(mu, sigma, f_best, cost, score_vec, budget, max_steps=10):
    """
    Simulate a budgeted greedy search.
    At each step, pick the candidate with highest score among those with cost <= remaining.
    Track the best value found (using mu as a proxy for the true value).
    """
    n = len(mu)
    remaining = budget
    visited = np.zeros(n, dtype=bool)
    best_val = f_best
    
    for _ in range(min(max_steps, n)):
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
    - weights: mix of EI, QOEI, UCB, cost_eff, cost-adjusted versions, and TS
    """
    n = len(mu)
    if n < 2:
        return 0.5, 1.0, (0.3, 0.2, 0.2, 0.1, 0.2, 0.0, 0.0, 0.0)
    
    # Budget estimate: total 30, initial 5 uniform points cost ~5*1.5=7.5
    # Remaining ~22.5. But cost varies. Let's use 20 as a safe estimate.
    budget = 20.0
    
    # Precompute all score components for each (q, beta) combination
    # We'll use a small grid
    q_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    beta_grid = [0.5, 1.0, 1.5, 2.0, 3.0]
    
    # Weight grid: 8 components now (added TS)
    # (ei, qoei, ucb, cost_eff, ei_cost, qoei_cost, ucb_cost, ts)
    w_grid = [
        (0.3, 0.2, 0.15, 0.05, 0.0, 0.0, 0.0, 0.3),  # TS-heavy
        (0.2, 0.2, 0.1, 0.05, 0.05, 0.0, 0.0, 0.4),  # More TS
        (0.25, 0.2, 0.15, 0.05, 0.05, 0.0, 0.0, 0.3),
        (0.2, 0.25, 0.15, 0.05, 0.05, 0.0, 0.0, 0.3),
        (0.3, 0.15, 0.15, 0.05, 0.0, 0.05, 0.0, 0.3),
        (0.25, 0.2, 0.1, 0.05, 0.05, 0.0, 0.0, 0.35),
        (0.2, 0.2, 0.15, 0.05, 0.05, 0.05, 0.0, 0.3),
        (0.15, 0.2, 0.15, 0.05, 0.05, 0.0, 0.0, 0.4),
        # Keep some old-style without TS for comparison
        (0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0),
        (0.2, 0.4, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0),
    ]
    
    best_regret = np.inf
    best_q = 0.5
    best_beta = 1.0
    best_w = (0.3, 0.2, 0.2, 0.1, 0.2, 0.0, 0.0, 0.0)
    
    # Precompute score components for each (q, beta)
    score_cache = {}
    for q in q_grid:
        for beta in beta_grid:
            key = (q, beta)
            comps = _score_components(mu, sigma, f_best, cost, q, beta)
            if comps is not None:
                score_cache[key] = comps
    
    # First pass: coarse search
    for w_tuple in w_grid:
        w_ei, w_qoei, w_ucb, w_ce, w_eic, w_qoc, w_ucbc, w_ts = w_tuple
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
                    w_ucbc * comps['ucb_cost'] +
                    w_ts * comps['ts']
                )
                score_vec = np.nan_to_num(score_vec, nan=0.0, posinf=0.0, neginf=0.0)
                
                best_val = _greedy_budget(mu, sigma, f_best, cost, score_vec, budget, max_steps=8)
                regret = f_best - best_val
                
                if regret < best_regret:
                    best_regret = regret
                    best_q = q
                    best_beta = beta
                    best_w = w_tuple
    
    # Second pass: fine-tune around best found
    # Create finer grid around best q and beta
    q_fine = np.clip(np.arange(best_q - 0.15, best_q + 0.15 + 1e-9, 0.05), 0.01, 0.99)
    beta_fine = np.clip(np.arange(best_beta - 0.5, best_beta + 0.5 + 1e-9, 0.25), 0.1, 5.0)
    
    # Fine-tune weights: perturb around best_w
    w_fine = [best_w]
    for i in range(len(best_w)):
        for delta in [-0.05, 0.05]:
            new_w = list(best_w)
            new_w[i] = max(0.0, best_w[i] + delta)
            # Renormalize
            s = sum(new_w)
            if s > 1e-9:
                new_w = tuple(w / s for w in new_w)
                w_fine.append(new_w)
    
    for w_tuple in w_fine:
        w_ei, w_qoei, w_ucb, w_ce, w_eic, w_qoc, w_ucbc, w_ts = w_tuple
        for q in q_fine:
            for beta in beta_fine:
                # Check if we need to compute new components
                key = (round(q, 4), round(beta, 4))
                if key not in score_cache:
                    comps = _score_components(mu, sigma, f_best, cost, float(q), float(beta))
                    if comps is None:
                        continue
                    score_cache[key] = comps
                else:
                    comps = score_cache[key]
                
                score_vec = (
                    w_ei * comps['ei'] +
                    w_qoei * comps['qoei'] +
                    w_ucb * comps['ucb'] +
                    w_ce * comps['cost_eff'] +
                    w_eic * comps['ei_cost'] +
                    w_qoc * comps['qoei_cost'] +
                    w_ucbc * comps['ucb_cost'] +
                    w_ts * comps['ts']
                )
                score_vec = np.nan_to_num(score_vec, nan=0.0, posinf=0.0, neginf=0.0)
                
                best_val = _greedy_budget(mu, sigma, f_best, cost, score_vec, budget, max_steps=10)
                regret = f_best - best_val
                
                if regret < best_regret:
                    best_regret = regret
                    best_q = float(q)
                    best_beta = float(beta)
                    best_w = w_tuple
    
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
        q, beta, w = 0.5, 1.0, (0.3, 0.2, 0.2, 0.1, 0.2, 0.0, 0.0, 0.0)
    
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
        w[6] * comps['ucb_cost'] +
        w[7] * comps['ts']
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
