"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_c4_bound(roots, m=12, n=1):
    """
    Compute the C4 upper bound from the Laguerre polynomial construction.
    P(t) = prod_i (t - x_i)^2 * L_m^(n)(t)
    C4 ~ 1 / ||P||_{L2(w)} where w(t) = t^n * exp(-t)
    """
    x = np.asarray(roots, dtype=np.float64)
    if x.shape != (m,) or np.any(x <= 0) or np.any(np.isnan(x)):
        return 1e10
    
    from scipy.special import eval_genlaguerre, roots_genlaguerre
    
    # Use Gauss-Laguerre quadrature with sufficient points
    n_quad = max(300, 6 * m + 100)
    tq, wq = roots_genlaguerre(n_quad, n)
    
    # Evaluate Laguerre polynomial at quadrature points
    L_vals = eval_genlaguerre(m, n, tq)
    
    # Evaluate the product prod_i (t - x_i)^2 at quadrature points
    # Use log-space for numerical stability
    log_prod_sq = np.zeros_like(tq)
    for xi in x:
        log_prod_sq += 2.0 * np.log(np.abs(tq - xi))
    
    prod_sq = np.exp(log_prod_sq)
    
    P_vals = prod_sq * L_vals
    
    # L2 norm squared under Laguerre weight
    norm_sq = np.sum(wq * P_vals ** 2)
    
    if norm_sq <= 0 or np.isnan(norm_sq) or not np.isfinite(norm_sq):
        return 1e10
    
    return 1.0 / np.sqrt(norm_sq)

def _log_root_objective(log_roots, m=12, n=1):
    """Objective in log-space: minimize C4 bound given log-roots."""
    if np.any(np.isnan(log_roots)) or np.any(np.isinf(log_roots)):
        return 1e10
    log_roots_clipped = np.clip(log_roots, -10, 30)
    roots = np.exp(log_roots_clipped)
    roots = np.sort(roots)
    if np.any(roots <= 0) or np.any(roots > 1e12):
        return 1e10
    return _compute_c4_bound(roots, m, n)

def construct():
    m = 12
    n = 1
    
    start_time = time.time()
    deadline = 600.0
    
    best_score = 1e10
    best_log_roots = None
    best_roots = None
    
    initial_log_roots = [
        np.log(np.array([(i + 1) ** 1.5 for i in range(m)])),
        np.log(np.array([(i + 1) ** 2.0 for i in range(m)])),
        np.log(np.array([(i + 1) ** 1.2 for i in range(m)])),
        np.log(np.array([3.0 + 2.0 * i for i in range(m)])),
        np.log(np.array([5.0 + 3.0 * i for i in range(m)])),
        np.log(np.array([2.0 + 8.0 * np.exp(-0.3 * i) for i in range(m)])),
        np.log(np.array([1.0 + 0.5 * i for i in range(6)] + [8.0 + 2.0 * i for i in range(6)])),
        np.log(np.array([0.5 + i * 0.8 for i in range(m)])),
    ]
    
    for i_init, x0 in enumerate(initial_log_roots):
        if time.time() - start_time > deadline - 60:
            break
        try:
            res = minimize(_log_root_objective, x0, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12,
                                   'maxls': 50})
            if res.fun < best_score and res.fun < 1e9 and np.all(np.isfinite(res.x)):
                best_score = res.fun
                best_log_roots = res.x.copy()
                best_roots = np.sort(np.exp(np.clip(res.x, -10, 30)))
        except Exception:
            continue
    
    if time.time() - start_time < deadline - 60 and best_log_roots is not None:
        try:
            res = minimize(_log_root_objective, best_log_roots, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True})
            if res.fun < best_score and res.fun < 1e9 and np.all(np.isfinite(res.x)):
                best_score = res.fun
                best_log_roots = res.x.copy()
                best_roots = np.sort(np.exp(np.clip(res.x, -10, 30)))
        except Exception:
            pass
    
    if time.time() - start_time < deadline - 30 and best_log_roots is not None:
        rng = np.random.default_rng(12345)
        n_hops = 30
        for hop in range(n_hops):
            if time.time() - start_time > deadline - 30:
                break
            perturbation = rng.normal(0, 1.0, size=m)
            perturbed_log = best_log_roots + perturbation
            try:
                res = minimize(_log_root_objective, perturbed_log, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_score and res.fun < 1e9 and np.all(np.isfinite(res.x)):
                    best_score = res.fun
                    best_log_roots = res.x.copy()
                    best_roots = np.sort(np.exp(np.clip(res.x, -10, 30)))
            except Exception:
                continue
    
    if time.time() - start_time < deadline - 90 and best_log_roots is not None:
        try:
            from scipy.optimize import differential_evolution
            bounds = [(-5.0, 25.0)] * m
            def de_objective(log_roots_flat):
                return _log_root_objective(log_roots_flat, m, n)
            res_de = differential_evolution(de_objective, bounds,
                                           maxiter=50, tol=1e-10,
                                           popsize=15, seed=99,
                                           polish=True,
                                           mutation=(0.3, 1.0),
                                           recombination=0.8)
            if res_de.fun < best_score and res_de.fun < 1e9 and np.all(np.isfinite(res_de.x)):
                best_score = res_de.fun
                best_log_roots = res_de.x.copy()
                best_roots = np.sort(np.exp(np.clip(res_de.x, -10, 30)))
        except Exception:
            pass
    
    if best_roots is None:
        best_roots = np.array([(i + 1) ** 1.5 for i in range(m)])
        best_roots = np.sort(best_roots)
    
    best_roots = np.maximum(best_roots, 1e-6)
    best_roots = np.sort(best_roots)
    
    return list(best_roots)
# EVOLVE-BLOCK-END
