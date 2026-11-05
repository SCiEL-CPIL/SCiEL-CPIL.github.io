"""Growth-index approximation for CPL dark-energy cosmologies.

Structural change: dual-space Chebyshev-projection pre-fit + minimax polish.
  1. Build reference f(a), D(a)/D(1) on a Chebyshev-spaced a-grid for a wider
     validation set.
  2. Stage A: weighted least-squares fit of the 6 coefficients using (i) f
     residuals, (ii) D residuals, and (iii) a smoothness penalty on dgamma/dln(a)
     to encourage equioscillation-friendly solutions.
  3. Stage B: trust-constr minimax polish on 100*max_rel_err with gamma-range
     and coefficient-bound constraints.
  4. Return the best coefficients via construct().
"""

import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.integrate import solve_ivp
import time

_START_TIME = time.time()
_TIME_LIMIT = 80.0  # safety margin before 90s kill

def _deadline_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

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
    """D(a)/D(1) = exp(-∫_{ln a}^0 f(e^t) dt)."""
    x = np.log(a)
    f_vals = _f_model(c, a, Om0, w0, wa)
    dx = np.diff(x)
    f_avg = 0.5 * (f_vals[1:] + f_vals[:-1])
    cum = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
    J = cum[-1] - cum
    return np.exp(-J)

def _gamma_dloga(c, a, Om0, w0, wa):
    """d gamma / d ln a  (analytic, for smoothness penalty)."""
    Om_m = _omega_m(a, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    c0, c1, c2, c3, c4, c5 = c
    dOm_dlna = -3.0 * Om_m * (Om_m - 1.0)
    dq_dlna = -dOm_dlna
    du_dlna = -wa * a
    return (c1 + 2.0 * c2 * q + c5 * u) * dq_dlna + (c3 + c5 * q) * du_dlna + c4 * wa * a

# ── Reference ODE solution ──────────────────────────────────────────────────

def _reference(a_grid, Om0, w0, wa):
    """Solve the growth ODE to get f_ref and D_ref."""
    x_min = np.log(a_grid[0])
    x_max = np.log(a_grid[-1])  # = 0
    
    def ode(x, y):
        a_val = np.exp(x)
        Om_m = _omega_m(a_val, Om0, w0, wa)
        w = w0 + wa * (1.0 - a_val)
        dOm_dx = -3.0 * Om_m * (Om_m - 1.0)
        D, F = y
        # D'' + [3/2 + (1+w) - 3/2 * dOm/dx] D' - 3/2 Om D = 0
        Dpp = -(1.5 + (1.0 + w) - 1.5 * dOm_dx) * F + 1.5 * Om_m * D
        return [F, Dpp]
    
    # Initial condition at small a: matter era D ~ a^(5/6)
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
        D = D / D[-1]  # normalize D(1)=1
        # f = F = D' = dD/dln a
        return F, D
    except Exception:
        # Fallback: constant gamma
        gam_ref = 0.55
        Om_m = _omega_m(a_grid, Om0, w0, wa)
        f_ref = Om_m ** gam_ref
        # Compute D via integration
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
    (0.32, -1.1, 0.2),
    (0.26, -0.7, -0.2),
    (0.30, -1.0, 0.8),
    (0.33, -0.95, -0.6),
]

# Chebyshev-spaced a-grid (dense in ln a for accurate D integration)
_N_A = 200
_k = np.arange(_N_A)
_t = np.cos(np.pi * _k / (_N_A - 1))  # 1 .. -1
_EVAL_A = np.exp(0.5 * (1.0 - _t) * (np.log(1.0) - np.log(5e-4)) + np.log(5e-4))
_EVAL_A = np.clip(_EVAL_A, 5e-4, 1.0)

# Pre-compute references
_REF = []
for (Om0, w0, wa) in _VALIDATION:
    if not _deadline_ok():
        break
    f_r, D_r = _reference(_EVAL_A, Om0, w0, wa)
    _REF.append((f_r, D_r, Om0, w0, wa))

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

# ── Chebyshev-projection least-squares pre-fit ──────────────────────────────

def _lsq_residuals(c, w_f=1.0, w_D=1.0, w_smooth=0.02):
    """Weighted residuals: f, D, and smoothness of dgamma/dlna."""
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return np.full(12, 1e3)

    res_parts = []
    for f_r, D_r, Om0, w0, wa in _REF:
        gam = _gamma(c, _EVAL_A, Om0, w0, wa)
        if np.any(gam < 0.1) or np.any(gam > 1.2):
            viol = np.maximum(0.0, 0.1 - gam) + np.maximum(0.0, gam - 1.2)
            res_parts.append(np.full(3, 5.0 * np.max(viol)))
            continue

        f_m = _f_model(c, _EVAL_A, Om0, w0, wa)
        D_m = _D_model(c, _EVAL_A, Om0, w0, wa)
        rel_f = (f_m - f_r) / np.maximum(np.abs(f_r), 1e-12)
        rel_D = (D_m - D_r) / np.maximum(np.abs(D_r), 1e-12)
        res_parts.append(w_f * rel_f)
        res_parts.append(w_D * rel_D)

        # Smoothness: penalize variation of dgamma/dlna (equioscillation proxy)
        gprime = _gamma_dloga(c, _EVAL_A, Om0, w0, wa)
        # Normalize by typical scale
        gprime_norm = gprime / np.maximum(np.abs(gprime).max(), 1e-8)
        res_parts.append(w_smooth * gprime_norm)

    return np.concatenate(res_parts)

# ── Optimization ────────────────────────────────────────────────────────────

def _optimize():
    best_c = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    best_err = _objective(best_c)

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage A: Chebyshev-projection least-squares pre-fit
    try:
        lsq_res = least_squares(_lsq_residuals, best_c, method='trf',
                                bounds=([-5.0] * 6, [5.0] * 6),
                                max_nfev=200, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if lsq_res.success or lsq_res.fun < best_err:
            cand = np.clip(lsq_res.x, -10, 10)
            err = _objective(cand)
            if err < best_err:
                best_err = err
                best_c = cand
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage B: trust-constr minimax polish with gamma-range + bound constraints
    try:
        def _minimax_obj(c):
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

        # Constraint: gamma in [0.1, 1.2] for all validation cosmologies
        def _gamma_lo(c):
            return np.array([
                np.min(_gamma(c, _EVAL_A, Om0, w0, wa)) - 0.1
                for (_, _, Om0, w0, wa) in _REF
            ])

        def _gamma_hi(c):
            return np.array([
                1.2 - np.max(_gamma(c, _EVAL_A, Om0, w0, wa))
                for (_, _, Om0, w0, wa) in _REF
            ])

        from scipy.optimize import LinearConstraint, NonlinearConstraint
        n_cons = len(_REF)
        cons = [
            NonlinearConstraint(_gamma_lo, 0.0, np.inf),
            NonlinearConstraint(_gamma_hi, 0.0, np.inf),
        ]
        bounds = [(-5.0, 5.0)] * 6

        res = minimize(_minimax_obj, best_c, method='trust-constr',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': 300, 'gtol': 1e-8, 'xtol': 1e-10})
        if res.fun < best_err:
            best_err = res.fun
            best_c = np.clip(res.x, -10, 10)
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage C: short Nelder-Mead polish (no constraints, rely on objective penalty)
    try:
        res = minimize(_objective, best_c, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10,
                                'adaptive': True})
        if res.fun < best_err:
            best_err = res.fun
            best_c = np.clip(res.x, -10, 10)
    except Exception:
        pass

    return np.clip(best_c, -10, 10)

_COEFFS = _optimize()

def construct():
    return list(_COEFFS)
