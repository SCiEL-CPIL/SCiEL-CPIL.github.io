"""Refined Cohn-Goncalves Laguerre construction optimizer."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import gamma

_START_TIME = time.time()
_DEADLINE = 580.0

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _objective(params):
    """Compute the upper bound C4 for given 12 positive root locations.
    
    C4 = integral_0^inf [prod_i (x - r_i)^4] e^{-x} dx / [prod_i r_i^4]
    """
    r = np.asarray(params, dtype=np.float64)
    if np.any(r <= 0) or np.any(~np.isfinite(r)):
        return 1e10
    
    m = len(r)
    
    # P(0) = prod_i r_i^2, so P(0)^2 = prod_i r_i^4
    log_P0_sq = 4.0 * np.sum(np.log(r))
    
    # Gauss-Laguerre quadrature
    n_quad = 150
    x_nodes, w_nodes = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute log of integrand: log(prod_i (x_j - r_i)^4 * e^{-x_j} * w_j)
    # = 4*sum_i log|x_j - r_i| - x_j + log(w_j)
    log_integrand = np.zeros(n_quad)
    for i in range(m):
        diff = x_nodes - r[i]
        log_integrand += 4.0 * np.log(np.maximum(np.abs(diff), 1e-300))
    log_integrand -= x_nodes
    log_integrand += np.log(w_nodes)
    
    # Sum in log-space to avoid overflow
    max_log = np.max(log_integrand)
    if max_log > 700:
        # Scale to prevent overflow
        integral = np.exp(max_log) * np.sum(np.exp(log_integrand - max_log))
    else:
        integral = np.sum(np.exp(log_integrand))
    
    if not np.isfinite(integral) or integral <= 0:
        return 1e10
    
    log_bound = np.log(integral) - log_P0_sq
    if log_bound > 700:
        return 1e10
    bound = np.exp(log_bound)
    
    if not np.isfinite(bound):
        return 1e10
    return bound

def _objective_log(params_log):
    """Objective in log-space."""
    r = np.exp(params_log)
    return _objective(r)

def construct():
    """Return 12 positive double-root locations minimizing the C4 upper bound."""
    t0 = time.time()
    
    # Multiple initial strategies
    candidates = []
    
    # Strategy 1: Geometric spacing
    r1 = np.geomspace(0.5, 30.0, 12)
    candidates.append(r1)
    
    # Strategy 2: Linear spacing near origin
    r2 = np.linspace(0.5, 20.0, 12)
    candidates.append(r2)
    
    # Strategy 3: Clustered near 1
    r3 = np.geomspace(0.3, 10.0, 12)
    candidates.append(r3)
    
    # Strategy 4: Seed program values
    r4 = np.array([4.0 * (i + 1) for i in range(12)])
    candidates.append(r4)
    
    best_r = None
    best_val = 1e20
    for r in candidates:
        val = _objective(r)
        if val < best_val:
            best_val = val
            best_r = r.copy()
    
    # Phase 1: Differential Evolution in log-space
    if _time_remaining() > 300:
        # Tighter bounds based on intuition
        bounds_log = [(np.log(0.05), np.log(100.0))] * 12
        try:
            de_result = differential_evolution(
                _objective_log,
                bounds=bounds_log,
                seed=42,
                maxiter=150,
                popsize=12,
                tol=1e-10,
                polish=False,
                init='latinhypercube',
                updating='immediate',
                mutation=(0.4, 0.8),
                recombination=0.8,
                seed=42
            )
            if de_result.fun < best_val:
                r_de = np.exp(de_result.x)
                val_de = _objective(r_de)
                if val_de < best_val and np.all(r_de > 0):
                    best_r = r_de
                    best_val = val_de
        except Exception:
            pass
    
    # Phase 2: L-BFGS in log-space
    if _time_remaining() > 60:
        try:
            res = minimize(
                _objective_log,
                np.log(best_r),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            if res.fun < best_val:
                r_lbg = np.exp(res.x)
                val_lbg = _objective(r_lbg)
                if val_lbg < best_val and np.all(r_lbg > 0):
                    best_r = r_lbg
                    best_val = val_lbg
        except Exception:
            pass
    
    # Phase 3: Direct L-BFGS
    if _time_remaining() > 30:
        try:
            res2 = minimize(
                _objective,
                best_r,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            if res2.fun < best_val and np.all(res2.x > 0):
                best_r = res2.x
                best_val = res2.fun
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead polish
    if _time_remaining() > 10:
        try:
            res3 = minimize(
                _objective,
                best_r,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res3.fun < best_val and np.all(res3.x > 0):
                best_r = res3.x
                best_val = res3.fun
        except Exception:
            pass
    
    # Final: sort and ensure positivity
    best_r = np.sort(np.maximum(best_r, 1e-10))
    
    return list(best_r)
