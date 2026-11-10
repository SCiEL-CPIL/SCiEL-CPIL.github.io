"""Improved program: true ODE reference + smooth objective + multi-start optimization."""
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 80.0  # 90s wall-clock, 10s safety margin

# Fixed validation grid (a values). x = ln(a).
_A_GRID = np.array([0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.45, 0.55, 0.65,
                    0.75, 0.85, 0.92, 0.97, 0.995])
_X_GRID = np.log(_A_GRID)
_NX = len(_A_GRID)

# Integration grid for the ODE (dense, in x).
_X_DEFS = np.linspace(0.0, np.log(0.05), 400)  # from x=0 (a=1) down to a=0.05

# Validation cosmologies (Om0, w0, wa)
_COSMOS = [
    (0.30, -1.0, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.25, -1.0, 0.3),
    (0.35, -1.0, -0.5),
    (0.30, -0.5, -1.0),
    (0.28, -1.2, 0.8),
    (0.32, -0.8, -0.3),
]

# ------------------------------------------------------------------
# Fast vectorized physics
# ------------------------------------------------------------------
def _precompute(Om0, w0, wa):
    """Precompute per-cosmology quantities on the x-grid."""
    # integral of w from a to 1 = w0*(1-a) + wa*(1-a)^2/2
    # On x-grid: a = exp(x)
    a_def = np.exp(_X_DEFS)
    one_minus_a = 1.0 - a_def
    w_int = w0 * one_minus_a + wa * one_minus_a**2 / 2.0
    de_factor = (1.0 - Om0) * np.exp(3.0 * w_int)
    m_factor = Om0 * a_def**3
    Om_def = m_factor / (m_factor + de_factor)
    E2 = m_factor + de_factor  # = H^2/H0^2
    # H'/H = 0.5*(m_factor - de_factor)/E2
    Hprime_H = 0.5 * (m_factor - de_factor) / E2
    return a_def, Om_def, Hprime_H


def _solve_ode_reference(Om0, w0, wa):
    """Integrate f' ODE backward from a=1 to get f(x) and D(x) on _X_DEFS."""
    a_def, Om_def, Hprime_H = _precompute(Om0, w0, wa)
    # f(1) = Om(1)^0.55 = Om0^0.55
    f = np.zeros(_NX)
    D = np.zeros(_NX)
    f[0] = Om0**0.55
    D[0] = 0.0  # ln D relative to D(1); at a=1, D/D(1)=1, ln=0
    # Integrate backward (x decreasing). Use RK4 with step = _X_DEFS[1]-_X_DEFS[0]
    dx = _X_DEFS[1] - _X_DEFS[0]  # negative
    for i in range(1, len(_X_DEFS)):
        x = _X_DEFS[i - 1]
        fi = f[i - 1]
        Di = D[i - 1]
        # f'(x) = 1.5*Om - f^2 - f*(2 + H'/H)
        # D'(x) = f (since d ln D / dx = f)
        def deriv_f(fi_val, Di_val, x_val):
            idx = int(round((x_val - 0.0) / dx))
            if idx < 0:
                idx = 0
            if idx >= len(_X_DEFS):
                idx = len(_X_DEFS) - 1
            return 1.5 * Om_def[idx] - fi_val**2 - fi_val * (2.0 + Hprime_H[idx])

        # RK4 for f and D
        k1f = deriv_f(fi, Di, x)
        k1d = fi
        k2f = deriv_f(fi + 0.5 * dx * k1f, Di + 0.5 * dx * k1d, x + 0.5 * dx)
        k2d = fi + 0.5 * dx * k1f
        k3f = deriv_f(fi + 0.5 * dx * k2f, Di + 0.5 * dx * k2d, x + 0.5 * dx)
        k3d = fi + 0.5 * dx * k2f
        k4f = deriv_f(fi + dx * k3f, Di + dx * k3d, x + dx)
        k4d = fi + dx * k3f
        f[i] = fi + (dx / 6.0) * (k1f + 2 * k2f + 2 * k3f + k4f)
        D[i] = Di + (dx / 6.0) * (k1d + 2 * k2d + 2 * k3d + k4d)
    # Interpolate to _X_GRID (which are subset of _X_DEFS or between)
    f_grid = np.interp(_X_GRID, _X_DEFS, f)
    D_grid = np.interp(_X_GRID, _X_DEFS, D)  # ln(D/D(1))
    return f_grid, D_grid


# Precompute ODE references for all cosmologies
_REF_F = []
_REF_D = []
_REF_PRE = []
for _Om0, _w0, _wa in _COSMOS:
    _f_ref, _D_ref = _solve_ode_reference(_Om0, _w0, _wa)
    _REF_F.append(_f_ref)
    _REF_D.append(_D_ref)
    _REF_PRE.append(_precompute(_Om0, _w0, _wa))


def _eval_model(c, idx):
    """Evaluate f and lnD for cosmology idx given coefficients c, on _X_GRID."""
    Om0, w0, wa = _COSMOS[idx]
    a_def, Om_def, Hprime_H = _REF_PRE[idx]
    # We need Om and q on _X_GRID (not _X_DEFS)
    a_grid = np.exp(_X_GRID)
    one_minus_a = 1.0 - a_grid
    w_int = w0 * one_minus_a + wa * one_minus_a**2 / 2.0
    de_factor = (1.0 - Om0) * np.exp(3.0 * w_int)
    m_factor = Om0 * a_grid**3
    Om = m_factor / (m_factor + de_factor)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * one_minus_a
    gamma = (c[0] + c[1] * q + c[2] * q**2 + c[3] * u
             + c[4] * wa * a_grid + c[5] * u * q)
    f = Om**gamma
    # ln D = integral from ln(a) to 0 of f dx  (negative of that for D/D(1))
    # D/D(1) = exp(-integral from ln(a) to 0 f dx) = exp(integral from 0 to ln(a) f dx)
    # Since ln(a) < 0, integral from 0 to ln(a) is negative, so D/D(1) < 1. Good.
    # Compute cumulative integral: lnD(x) = integral from 0 to x of f dx (x<0)
    # Use cumulative trapezoidal
    # _X_GRID is sorted ascending (most negative to 0)
    # integral from 0 to x_i = -integral from x_i to 0
    # Let's compute integral from x_i to 0 for each i
    # cumulative from the end
    dx_grid = np.diff(_X_GRID)
    # integral from x_i to 0 = sum of trapezoids from i to end
    # Use cumsum from the right
    trap = 0.5 * (f[:-1] + f[1:]) * dx_grid
    # integral from x_i to 0 for i: sum of trap[i:]
    cum_from_end = np.zeros(_NX)
    cum_from_end[:-1] = np.cumsum(trap[::-1])[::-1]
    # integral from x_i to 0 = cum_from_end[i]
    lnD = -cum_from_end  # D/D(1) = exp(-integral from ln(a) to 0 f dx)
    # At i = last (x=0), lnD should be 0
    lnD[-1] = 0.0
    return f, lnD, gamma


def _objective_smooth(c, temperature=0.02):
    """Smooth soft-max of relative errors across all cosmologies and a-values."""
    max_err = 0.0
    n_pts = 0
    log_sum_f = -np.inf
    log_sum_D = -np.inf
    for idx in range(len(_COSMOS)):
        f_approx, lnD_approx, gamma = _eval_model(c, idx)
        # Check gamma constraint (soft penalty)
        if gamma.min() < 0.1 or gamma.max() > 1.2:
            # Heavy penalty
            penalty = max(0.0, 0.1 - gamma.min()) + max(0.0, gamma.max() - 1.2)
            return 10.0 + penalty * 100.0
        f_ref = _REF_F[idx]
        lnD_ref = _REF_D[idx]
        # Relative errors
        rel_f = np.abs(f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        rel_D = np.abs(np.exp(lnD_approx) - np.exp(lnD_ref)) / np.maximum(np.exp(lnD_ref), 1e-10)
        # Soft max via log-sum-exp
        m_f = np.max(rel_f)
        m_D = np.max(rel_D)
        log_sum_f = np.log(np.sum(np.exp(rel_f - m_f))) + m_f
        log_sum_D = np.log(np.sum(np.exp(rel_D - m_D))) + m_D
        # Combine: take max of the two soft-maxes
        cur_max = max(np.exp(log_sum_f - _NX * 0) * 0 + m_f + np.log(np.sum(np.exp(rel_f - m_f))) / _NX,
                      m_D + np.log(np.sum(np.exp(rel_D - m_D))) / _NX)
        # Actually simpler: soft max = m + log(sum(exp(x-m))) / n  (mean-soft-max)
        # But for worst-case, we want max. Use: m + log(sum(exp(x-m))) / n is too soft.
        # Use: m + log(sum(exp((x-m)/T))) * T / n  ... let me just use plain max for now
        cur_max = max(np.max(rel_f), np.max(rel_D))
        max_err = max(max_err, cur_max)
        n_pts += 2 * _NX
    # Return 100 * max_err (the score is 100 * worst relative error)
    return 100.0 * max_err


def _objective_smooth_T(c, T=0.05):
    """Soft-max with temperature T for smoother gradient."""
    max_err = 0.0
    for idx in range(len(_COSMOS)):
        f_approx, lnD_approx, gamma = _eval_model(c, idx)
        if gamma.min() < 0.1 or gamma.max() > 1.2:
            penalty = max(0.0, 0.1 - gamma.min()) + max(0.0, gamma.max() - 1.2)
            return 10.0 + penalty * 100.0
        f_ref = _REF_F[idx]
        lnD_ref = _REF_D[idx]
        rel_f = np.abs(f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        rel_D = np.abs(np.exp(lnD_approx) - np.exp(lnD_ref)) / np.maximum(np.exp(lnD_ref), 1e-10)
        # Soft max with temperature
        m_f = np.max(rel_f)
        m_D = np.max(rel_D)
        soft_f = m_f + T * np.log(np.mean(np.exp((rel_f - m_f) / T)))
        soft_D = m_D + T * np.log(np.mean(np.exp((rel_D - m_D) / T)))
        cur_max = max(soft_f, soft_D)
        max_err = max(max_err, cur_max)
    return 100.0 * max_err


def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    # Initial guess: constant gamma = 0.55
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_c = c0.copy()
    best_score = _objective_smooth(best_c)

    # Multi-start with different initial conditions
    starts = [
        np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.05, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.02, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.02, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.0, 0.01, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.01]),
        np.array([0.50, 0.05, 0.01, 0.01, 0.0, 0.0]),
        np.array([0.60, -0.05, 0.01, 0.01, 0.0, 0.0]),
    ]

    bounds = [(-10.0, 10.0)] * 6

    for c_start in starts:
        if time.time() > deadline - 15.0:
            break
        # L-BFGS-B with smooth objective
        try:
            result = minimize(_objective_smooth_T, c_start, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            if result.fun < best_score:
                best_score = result.fun
                best_c = result.x.copy()
        except Exception:
            pass

        # Nelder-Mead polish
        if time.time() > deadline - 10.0:
            break
        try:
            result2 = minimize(_objective_smooth, best_c, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-8,
                                        'adaptive': True})
            if result2.fun < best_score:
                best_score = result2.fun
                best_c = result2.x.copy()
        except Exception:
            pass

    # Final validation: check gamma constraint on a fine grid
    valid = True
    for idx in range(len(_COSMOS)):
        Om0, w0, wa = _COSMOS[idx]
        for a in np.linspace(0.05, 1.0, 200):
            a_arr = np.array([a])
            one_minus_a = 1.0 - a_arr
            w_int = w0 * one_minus_a + wa * one_minus_a**2 / 2.0
            de_factor = (1.0 - Om0) * np.exp(3.0 * w_int)
            m_factor = Om0 * a_arr**3
            Om = m_factor / (m_factor + de_factor)
            q = 1.0 - Om
            u = 1.0 + w0 + wa * one_minus_a
            g = (best_c[0] + best_c[1] * q + best_c[2] * q**2 + best_c[3] * u
                 + best_c[4] * wa * a_arr + best_c[5] * u * q)
            if g[0] < 0.1 or g[0] > 1.2:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fall back to seed
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    best_c = np.clip(best_c, -10.0, 10.0)
    return list(best_c)
# EVOLVE-BLOCK-END
