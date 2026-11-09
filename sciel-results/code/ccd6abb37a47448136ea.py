import numpy as np
import time

# ---- Deadline guard (shared across the process; evaluator kills at 330s) ----
_START_TIME = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 20.0

# ---- Constants of the task ----
DIM = 4
RANGE_LOW = -4.0
RANGE_HIGH = 5.0
SPAN = RANGE_HIGH - RANGE_LOW          # 9.0
COST_BUDGET = 30.0
N_INIT = 5
N_CAND = 2000

# ---- Acquisition form (cost-normalized POI + deterministic regret lower bound) ----
def _compute_scores(mu, sigma, cost, f_best, p):
    eps = 1e-9
    safe_sigma = np.where(sigma < eps, eps, sigma)
    safe_sigma = np.where(np.isfinite(safe_sigma), safe_sigma, eps)
    safe_cost = np.where(cost < eps, eps, cost)
    safe_cost = np.where(np.isfinite(safe_cost), safe_cost, eps)
    mu_safe = np.where(np.isfinite(mu), mu, f_best)

    # deterministic regret lower bound: expected improvement if we could
    # reach mu - beta*sigma (a conservative lower bound on the true value)
    bound = f_best - (mu_safe + p["beta"] * safe_sigma)

    score = bound / safe_cost
    return np.where(np.isfinite(score), score, -1e10)

# ---- Public entry point (name & return interface preserved) ----
_PARAMS = None
def _default_params():
    return {"beta": 0.5}

def acquisition(mu, sigma, f_best, cost):
    global _PARAMS
    if _PARAMS is None:
        _PARAMS = _default_params()
    # hard deadline guard -> fall back to plain cost-normalized bound
    if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        cost_arr = np.asarray(cost, dtype=np.float64)
        mu_arr = np.asarray(mu, dtype=np.float64)
        return (f_best - mu_arr) / np.where(cost_arr < 1e-9, 1e-9, cost_arr)
    return _compute_scores(np.asarray(mu, dtype=np.float64),
                           np.asarray(sigma, dtype=np.float64),
                           np.asarray(cost, dtype=np.float64),
                           float(f_best), _PARAMS)
