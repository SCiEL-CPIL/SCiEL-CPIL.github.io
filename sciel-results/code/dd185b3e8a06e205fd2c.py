"""Growth-index approximation for CPL dark-energy cosmologies.

Smooth epigraph minimax via log-sum-exp:
  1. Dense Chebyshev grid in log(a) for equioscillation.
  2. Stage A: Linear LSQ pre-fit in log-space for initialization.
  3. Stage B: SLSQP on smooth log-sum-exp epigraph of relative errors.
  4. Return optimized 6 coefficients via construct().
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

_START_TIME = time.time()
_TIME_LIMIT = 72.0  # safety margin before 90s kill

def _deadline_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

def _chebyshev_nodes(n):
    """Return n Chebyshev nodes in [-1, 1]."""
    k = np.arange(n)
    return np.cos(np.pi * (2*k + 1) / (2*n))

# ── Physics ─────────────────────────────────────────────────────────────────

def _omega_m(a, Om0, w0, wa):
    """Matter density parameter Ω_m(a) for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    a = np.clip(a, 1e-10, 1e6)
    a3 = a ** 3
    de_ratio = a3 ** (-(1.0 + w0 + wa)) * np.exp(-3.0 * wa * (a - 1.0))
    om_m = Om0 / (Om0 + (1.0 - Om0) * de_ratio)
    return om_m

def _gamma(c, a, Om0, w0, wa):
    """Compact 6-coefficient gamma ansatz."""
    Om_m = _omega_m(a, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    c0, c1, c2, c3, c4, c5 = c
    return c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q

def _f_model(c, a, Om0, w0, wa):
    """f(a) = Ω_m(a)^gamma(a) from the ansatz."""
    Om_m = _omega_m(a, Om0, w0, wa)
    gam = _gamma(c, a, Om0, w0, wa)
    Om_m = np.clip(Om_m, 1e-15, 1.0)
    return Om_m ** gam

def _D_model(c, a, Om0, w0, wa):
    """D(a)/D(1) = exp(-∫_{ln a}^0 f(e^x) dx)"""
    x = np.log(a)
    f_vals = _f_model(c, a, Om0, w0, wa)
    dx = np.diff(x)
    f_avg = 0.5 * (f_vals[1:] + f_vals[:-1])
    cum = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
    J = cum[-1] - cum
    return np.exp(-J)

# ── Reference ODE solution ──────────────────────────────────────────────────

def _reference(a_grid, Om0, w0, wa):
    """Solve the growth ODE to get f_ref and D_ref."""
    x_min = np.log(a_grid[0])
    x_max = np.log(a_grid[-1])
    
    def ode(x, y):
        a_val = np.exp(x)
        Om_m = _omega_m(a_val, Om0, w0, wa)
        w = w0 + wa * (1.0 - a_val)
        dOm_dx = -3.0 * Om_m * (Om_m - 1.0)
        D, F = y
        Dpp = -(1.5 + (1.0 + w) - 1.5 * dOm_dx) * F + 1.5 * Om_m * D
        return [F, Dpp]
    
    a0 = a_grid[0]
    D0 = a0 ** (5.0 / 6.0)
    F0 = (5.0 / 6.0) * a0 ** (-1.0 / 6.0)
    
    try:
        sol = solve_ivp(ode, [x_min, x_max], [D0, F0],
                        t_eval=a_grid, max_step=0.001, rtol=1e-10, atol=1e-12)
        if not sol.success:
            raise RuntimeError("ODE failed")
        D = sol.y[0]
        F = sol.y[1]
        D = D / D[-1]
        return F, D
    except Exception:
        gam_ref = 0.55
        Om_m = _omega_m(a_grid, Om0, w0, wa)
        f_ref = Om_m ** gam_ref
        x = np.log(a_grid)
        dx = np.diff(x)
        f_avg = 0.5 * (f_ref[1:] + f_ref[:-1])
        cum = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
        J = cum[-1] - cum
        D = np.exp(-J)
        D = D / D[-1]
        return f_ref, D

# ── Validation cosmologies ──────────────────────────────────────────────────

_VALIDATION = [
    (0.30, -1.0, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.30, -0.5, 0.0),
    (0.30, -0.8, 0.3),
    (0.25, -1.0, 0.5),
    (0.35, -1.0, -0.3),
    (0.30, -1.2, 0.4),
    (0.30, -0.9, -0.4),
    (0.28, -1.0, 1.0),
    (0.30, -1.0, -1.0),
    (0.27, -0.9, 0.6),
]

# Dense Chebyshev grid
_N_POINTS = 512
_cheb_x = _chebyshev_nodes(_N_POINTS)
_log_a_min = np.log(3e-4)
_log_a_max = 0.0
_EVAL_A = np.exp(_log_a_min + (_cheb_x + 1) / 2 * (_log_a_max - _log_a_min))
_EVAL_A = np.sort(_EVAL_A)

# Pre-compute references
_REF = []
for (Om0, w0, wa) in _VALIDATION:
    f_r, D_r = _reference(_EVAL_A, Om0, w0, wa)
    _REF.append((f_r, D_r, Om0, w0, wa))

# Pre-compute features for all models
_FEAT_CACHE = []
for f_r, D_r, Om0, w0, wa in _REF:
    Om_m = _omega_m(_EVAL_A, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - _EVAL_A)
    phi = np.column_stack([
        np.ones_like(q),
        q,
        q ** 2,
        u,
        wa * _EVAL_A,
        u * q
    ])
    ln_Om = np.log(np.clip(Om_m, 1e-15, 1.0))
    _FEAT_CACHE.append((phi, ln_Om, f_r, D_r, Om0, w0, wa))

# ── Objectives ──────────────────────────────────────────────────────────────

def _max_rel_error(c):
    """Compute the maximum relative error across all models and a-values."""
    c = np.asarray(c, dtype=float)
    max_err = 0.0
    for phi, ln_Om, f_r, D_r, Om0, w0, wa in _FEAT_CACHE:
        gam = phi @ c
        if np.any(gam < 0.1) or np.any(gam > 1.2):
            viol = np.maximum(0.0, 0.1 - gam) + np.maximum(0.0, gam - 1.2)
            max_err = max(max_err, 5.0 * np.max(viol))
            continue
        
        Om_m = np.exp(ln_Om)
        f_m = Om_m ** gam
        D_m = _D_model(c, _EVAL_A, Om0, w0, wa)
        
        rel_f = np.abs(f_m - f_r) / np.maximum(np.abs(f_r), 1e-12)
        rel_D = np.abs(D_m - D_r) / np.maximum(np.abs(D_r), 1e-12)
        
        max_err = max(max_err, np.max(rel_f), np.max(rel_D))
    
    return 100.0 * max_err

def _smooth_epigraph_objective(c, beta=50.0):
    """Smooth approximation of max relative error via log-sum-exp."""
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6
    
    # Collect all relative errors
    errors = []
    for phi, ln_Om, f_r, D_r, Om0, w0, wa in _FEAT_CACHE:
        gam = phi @ c
        # Gamma constraint penalty
        gam_pen = np.maximum(0.0, 0.1 - gam) + np.maximum(0.0, gam - 1.2)
        if np.any(gam_pen > 0):
            return 1e6
        
        Om_m = np.exp(ln_Om)
        f_m = Om_m ** gam
        D_m = _D_model(c, _EVAL_A, Om0, w0, wa)
        
        rel_f = np.abs(f_m - f_r) / np.maximum(np.abs(f_r), 1e-12)
        rel_D = np.abs(D_m - D_r) / np.maximum(np.abs(D_r), 1e-12)
        
        errors.extend(rel_f.tolist())
        errors.extend(rel_D.tolist())
    
    err_arr = np.array(errors)
    
    # Log-sum-exp for smooth max
    max_e = np.max(err_arr)
    if max_e > 10:
        return 1e6
    log_sum = np.log(np.sum(np.exp(beta * (err_arr - max_e)))) / beta + max_e
    return 100.0 * log_sum

def _lsq_logspace_fit():
    """Stage A: Linear LSQ fit in log-space."""
    try:
        rows = []
        targets = []
        for phi, ln_Om, f_r, D_r, Om0, w0, wa in _FEAT_CACHE:
            ln_f = np.log(np.clip(f_r, 1e-15, 1.0))
            mask = np.abs(ln_Om) > 0.01
            if np.sum(mask) > 10:
                rows.append(phi[mask])
                targets.append(ln_f[mask] / ln_Om[mask])
        
        A = np.vstack(rows)
        b = np.concatenate(targets)
        c_lsq, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        c_lsq = np.clip(c_lsq, -10, 10)
        
        # Check gamma constraint
        max_viol = 0.0
        for phi, ln_Om, f_r, D_r, Om0, w0, wa in _FEAT_CACHE:
            gam = phi @ c_lsq
            viol = np.maximum(0.0, 0.1 - gam) + np.maximum(0.0, gam - 1.2)
            max_viol = max(max_viol, np.max(viol))
        
        if max_viol > 0.05:
            return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
        
        return c_lsq
    except Exception:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

def _optimize():
    # Stage A: Log-space LSQ
    c_lsq = _lsq_logspace_fit()
    best_c = c_lsq.copy()
    best_err = _max_rel_error(best_c)
    
    seed_c = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    seed_err = _max_rel_error(seed_c)
    if seed_err < best_err:
        best_c = seed_c.copy()
        best_err = seed_err
    
    if not _deadline_ok():
        return np.clip(best_c, -10, 10)
    
    # Stage B: SLSQP on smooth epigraph
    bounds = [(-3, 3)] * 6
    try:
        res = minimize(_smooth_epigraph_objective, best_c, method='SLSQP',
                       bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12})
        if res.fun < best_err:
            true_err = _max_rel_error(res.x)
            if true_err < best_err:
                best_err = true_err
                best_c = res.x.copy()
    except Exception:
        pass
    
    if not _deadline_ok():
        return np.clip(best_c, -10, 10)
    
    # Stage C: Refine with higher beta (sharper max approximation)
    try:
        res = minimize(lambda c: _smooth_epigraph_objective(c, beta=200.0), best_c,
                       method='SLSQP', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12})
        if res.fun < best_err:
            true_err = _max_rel_error(res.x)
            if true_err < best_err:
                best_err = true_err
                best_c = res.x.copy()
    except Exception:
        pass
    
    if not _deadline_ok():
        return np.clip(best_c, -10, 10)
    
    # Stage D: Final Nelder-Mead polish on true max
    try:
        res = minimize(_max_rel_error, best_c, method='Nelder-Mead',
                       options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        if res.fun < best_err:
            best_err = res.fun
            best_c = res.x.copy()
    except Exception:
        pass
    
    return np.clip(best_c, -10, 10)

_COEFFS = _optimize()

def construct():
    return list(_COEFFS)
