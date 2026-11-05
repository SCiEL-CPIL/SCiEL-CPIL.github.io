"""Growth-index approximation for CPL dark-energy cosmologies.

Spectral residual strategy:
  1. Pre-compute reference ODE solutions on Chebyshev grid.
  2. Build linear LSQ system for gamma from ln(f) and ln(D) residuals.
  3. Solve for 6 coefficients via least squares.
  4. Polish with Nelder-Mead on exact minimax objective.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

_START_TIME = time.time()
_TIME_LIMIT = 75.0

def _deadline_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

def _chebyshev_nodes(n):
    k = np.arange(n)
    return np.cos(np.pi * (2*k + 1) / (2*n))

# ── Physics ─────────────────────────────────────────────────────────────────

def _omega_m(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    a = np.clip(a, 1e-10, 1e6)
    a3 = a ** 3
    de_ratio = a3 ** (-(1.0 + w0 + wa)) * np.exp(-3.0 * wa * (a - 1.0))
    om_m = Om0 / (Om0 + (1.0 - Om0) * de_ratio)
    return om_m

def _gamma(c, a, Om0, w0, wa):
    Om_m = _omega_m(a, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    c0, c1, c2, c3, c4, c5 = c
    return c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q

def _f_model(c, a, Om0, w0, wa):
    Om_m = _omega_m(a, Om0, w0, wa)
    gam = _gamma(c, a, Om0, w0, wa)
    Om_m = np.clip(Om_m, 1e-15, 1.0)
    return Om_m ** gam

def _D_model(c, a, Om0, w0, wa):
    x = np.log(a)
    f_vals = _f_model(c, a, Om0, w0, wa)
    dx = np.diff(x)
    f_avg = 0.5 * (f_vals[1:] + f_vals[:-1])
    cum = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
    J = cum[-1] - cum
    return np.exp(-J)

# ── Reference ODE solution ──────────────────────────────────────────────────

def _reference(a_grid, Om0, w0, wa):
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
                        t_eval=a_grid, max_step=0.002, rtol=1e-9, atol=1e-11)
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

_N_POINTS = 256
_cheb_x = _chebyshev_nodes(_N_POINTS)
_log_a_min = np.log(5e-4)
_log_a_max = 0.0
_EVAL_A = np.exp(_log_a_min + (_cheb_x + 1) / 2 * (_log_a_max - _log_a_min))
_EVAL_A = np.sort(_EVAL_A)

_REF = []
for (Om0, w0, wa) in _VALIDATION:
    f_r, D_r = _reference(_EVAL_A, Om0, w0, wa)
    _REF.append((f_r, D_r, Om0, w0, wa))

# ── Spectral residual: linear LSQ on ln(f) and ln(D) ───────────────────────

def _build_spectral_system():
    """Build linear system A c = b where c = [c0..c5].
    
    For f = Om^gamma: ln(f) = gamma * ln(Om)
    So d(ln f)/d(ln Om) = gamma.
    
    We regress gamma on the basis features [1, q, q^2, u, wa*a, u*q].
    Additionally, we use ln(D) residuals to improve the fit.
    
    Since ln(D)/D(1) = -∫f dx, and f = Om^gamma,
    small errors in gamma propagate to D with weight ~ ln(Om) * integral length.
    
    Strategy: weight the gamma regression by |ln(Om)| to emphasize
    regions where gamma errors most affect D.
    """
    A_rows = []
    b_vals = []
    
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
        
        # Target: gamma_ref = d(ln f)/d(ln Om)
        ln_f = np.log(np.clip(f_r, 1e-15, 1.0))
        ln_Om = np.log(np.clip(Om_m, 1e-15, 1.0))
        
        # Central difference for gamma
        dlnf_dlnom = np.zeros_like(ln_f)
        for i in range(len(ln_f)):
            if i > 0 and i < len(ln_f) - 1:
                denom = ln_Om[i+1] - ln_Om[i-1]
                if abs(denom) > 1e-10:
                    dlnf_dlnom[i] = (ln_f[i+1] - ln_f[i-1]) / denom
                else:
                    dlnf_dlnom[i] = 0.55
            elif i == 0:
                denom = ln_Om[1] - ln_Om[0]
                if abs(denom) > 1e-10:
                    dlnf_dlnom[i] = (ln_f[1] - ln_f[0]) / denom
                else:
                    dlnf_dlnom[i] = 0.55
            else:
                denom = ln_Om[-1] - ln_Om[-2]
                if abs(denom) > 1e-10:
                    dlnf_dlnom[i] = (ln_f[-1] - ln_f[-2]) / denom
                else:
                    dlnf_dlnom[i] = 0.55
        
        # Weight by |ln(Om)| to emphasize D sensitivity
        weights = np.abs(ln_Om) + 0.1
        
        A_rows.append(phi * weights[:, np.newaxis])
        b_vals.append(dlnf_dlnom * weights)
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    return A, b

def _lsq_fit():
    """Solve the linear LSQ system for coefficients."""
    try:
        A, b = _build_spectral_system()
        if A is None or len(A) == 0:
            return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
        
        c_lsq, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        c_lsq = np.clip(c_lsq, -10, 10)
        
        # Check gamma constraints
        max_viol = 0.0
        for f_r, D_r, Om0, w0, wa in _REF:
            gam = _gamma(c_lsq, _EVAL_A, Om0, w0, wa)
            viol = np.maximum(0.0, 0.1 - gam) + np.maximum(0.0, gam - 1.2)
            max_viol = max(max_viol, np.max(viol))
        
        if max_viol > 0.02:
            # Fallback to seed
            return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
        
        return c_lsq
    except Exception:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

# ── Objective ───────────────────────────────────────────────────────────────

def _objective(c):
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6
    
    max_err = 0.0
    for f_r, D_r, Om0, w0, wa in _REF:
        gam = _gamma(c, _EVAL_A, Om0, w0, wa)
        if np.any(gam < 0.1) or np.any(gam > 1.2):
            viol = np.maximum(0.0, 0.1 - gam) + np.maximum(0.0, gam - 1.2)
            max_err = max(max_err, 5.0 * np.max(viol))
            continue
        
        f_m = _f_model(c, _EVAL_A, Om0, w0, wa)
        D_m = _D_model(c, _EVAL_A, Om0, w0, wa)
        
        rel_f = np.abs(f_m - f_r) / np.maximum(np.abs(f_r), 1e-12)
        rel_D = np.abs(D_m - D_r) / np.maximum(np.abs(D_r), 1e-12)
        
        max_err = max(max_err, np.max(rel_f), np.max(rel_D))
    
    return 100.0 * max_err

def _smooth_objective(c, temperature=0.005):
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6
    
    all_errors = []
    for f_r, D_r, Om0, w0, wa in _REF:
        gam = _gamma(c, _EVAL_A, Om0, w0, wa)
        if np.any(gam < 0.1) or np.any(gam > 1.2):
            viol = np.maximum(0.0, 0.1 - gam) + np.maximum(0.0, gam - 1.2)
            all_errors.append(5.0 * viol)
            continue
        
        f_m = _f_model(c, _EVAL_A, Om0, w0, wa)
        D_m = _D_model(c, _EVAL_A, Om0, w0, wa)
        
        rel_f = np.abs(f_m - f_r) / np.maximum(np.abs(f_r), 1e-12)
        rel_D = np.abs(D_m - D_r) / np.maximum(np.abs(D_r), 1e-12)
        
        all_errors.append(rel_f)
        all_errors.append(rel_D)
    
    all_errors = np.concatenate(all_errors)
    if len(all_errors) == 0:
        return 1e6
    
    scaled = all_errors / temperature
    max_s = np.max(scaled)
    if max_s > 700:
        log_sum_exp = max_s
    else:
        log_sum_exp = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    
    return 100.0 * temperature * log_sum_exp

# ── Optimization ────────────────────────────────────────────────────────────

def _optimize():
    # Stage 1: Linear LSQ fit (spectral residual)
    c_lsq = _lsq_fit()
    best_c = c_lsq.copy()
    best_err = _objective(best_c)
    
    # Compare with seed
    seed_c = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    seed_err = _objective(seed_c)
    if seed_err < best_err:
        best_c = seed_c.copy()
        best_err = seed_err
    
    if not _deadline_ok():
        return np.clip(best_c, -10, 10)
    
    # Stage 2: L-BFGS-B on smooth objective from LSQ start
    try:
        res = minimize(_smooth_objective, best_c, method='L-BFGS-B',
                       bounds=[(-4, 4)] * 6,
                       options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
        fun = _objective(res.x)
        if fun < best_err:
            best_err = fun
            best_c = res.x.copy()
    except Exception:
        pass
    
    if not _deadline_ok():
        return np.clip(best_c, -10, 10)
    
    # Stage 3: Nelder-Mead on smooth objective
    try:
        res = minimize(_smooth_objective, best_c, method='Nelder-Mead',
                       options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
        fun = _objective(res.x)
        if fun < best_err:
            best_err = fun
            best_c = res.x.copy()
    except Exception:
        pass
    
    if not _deadline_ok():
        return np.clip(best_c, -10, 10)
    
    # Stage 4: Exact minimax Nelder-Mead
    try:
        res = minimize(_objective, best_c, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True})
        if res.fun < best_err:
            best_err = res.fun
            best_c = res.x.copy()
    except Exception:
        pass
    
    if not _deadline_ok():
        return np.clip(best_c, -10, 10)
    
    # Stage 5: Multi-start refinement
    rng = np.random.default_rng(123)
    for i in range(15):
        if not _deadline_ok():
            break
        scale = 0.02 + 0.02 * (1.0 - i / 15.0)
        pert = rng.normal(0, scale, 6)
        x0 = np.clip(best_c + pert, -4, 4)
        try:
            res = minimize(_smooth_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            fun = _objective(res.x)
            if fun < best_err:
                best_err = fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Final polish
    if _deadline_ok():
        try:
            res = minimize(_objective, best_c, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_err:
                best_err = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    return np.clip(best_c, -10, 10)

_COEFFS = _optimize()

def construct():
    return list(_COEFFS)
