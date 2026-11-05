"""Growth-index approximation for CPL dark-energy cosmologies.

Dual-space Chebyshev projection strategy:
  1. Map (q, u) to a normalized Chebyshev grid for equioscillation properties.
  2. Stage A: Linear least-squares pre-fit in log-space (ln f = gamma * ln Om).
  3. Stage B: Minimax polish with derivative-aware objective for smoothness.
  4. Return optimized 6 coefficients via construct().
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

_START_TIME = time.time()
_TIME_LIMIT = 75.0  # safety margin before 90s kill

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
    # Ω_m(a) = Om0 / [Om0 + (1-Om0) * (a/a0)^(-3(1+w0+wa)) * exp(-3*wa*(a-1)/a0)]
    # With a0=1:
    de_ratio = a3 ** (-(1.0 + w0 + wa)) * np.exp(-3.0 * wa * (a - 1.0))
    # Actually the standard formula:
    # E^2 = Om0 * a^-3 + (1-Om0) * a^-3(1+w0+wa) * exp(3*wa*(1-a))
    # Ω_m = Om0 * a^-3 / E^2
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
    """D(a)/D(1) = exp(-∫_{ln a}^0 f(e^x) dx) = exp(∫_0^{ln a} f(e^x) dx)"""
    # We need ∫ from ln(a) to 0 of f(exp(x)) dx = -∫ from 0 to ln(a) of f(exp(x)) dx
    # So D/D(1) = exp(-∫_{ln a}^0 ...) = exp(∫_0^{ln a} ...)
    # For a < 1, ln a < 0, so ∫_0^{ln a} is negative of ∫_{ln a}^0.
    # Let's compute J(a) = ∫_{ln a}^0 f(e^t) dt  (positive for a<1)
    # Then D/D(1) = exp(-J(a))
    
    x = np.log(a)
    # We want for each x_i: J_i = ∫_{x_i}^{0} f(e^t) dt
    # Use cumulative integration from x[0] to x[i], then subtract from total.
    
    f_vals = _f_model(c, a, Om0, w0, wa)
    # Cumulative trapezoid from x[0] to x[i]
    dx = np.diff(x)
    f_avg = 0.5 * (f_vals[1:] + f_vals[:-1])
    cum = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
    # cum[i] = ∫_{x[0]}^{x[i]} f dt
    # Total = cum[-1] = ∫_{x[0]}^{0} f dt  (since x[-1]=0)
    # J_i = ∫_{x[i]}^{0} f dt = cum[-1] - cum[i]
    J = cum[-1] - cum
    return np.exp(-J)

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
]

# Use Chebyshev-spaced a values for better equioscillation
_N_POINTS = 256
_cheb_x = _chebyshev_nodes(_N_POINTS)  # in [-1, 1]
# Map to log(a) space: log(a) from log(5e-4) to 0
_log_a_min = np.log(5e-4)
_log_a_max = 0.0
_EVAL_A = np.exp(_log_a_min + (_cheb_x + 1) / 2 * (_log_a_max - _log_a_min))
_EVAL_A = np.sort(_EVAL_A)  # ensure ascending

# Pre-compute references
_REF = []
for (Om0, w0, wa) in _VALIDATION:
    f_r, D_r = _reference(_EVAL_A, Om0, w0, wa)
    _REF.append((f_r, D_r, Om0, w0, wa))

# ── Chebyshev-based feature matrix for log-space linear fit ────────────────
# Build the design matrix for gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
# We want to solve: ln(f_ref) ≈ gamma * ln(Om) = (sum c_i * phi_i) * ln(Om)
# So: ln(f_ref) / ln(Om) ≈ sum c_i * phi_i  (when ln(Om) != 0)
# This is linear in c!

def _build_lsq_matrix():
    """Build the linear least-squares matrix for log-space fit."""
    rows = []
    targets = []
    for f_r, D_r, Om0, w0, wa in _REF:
        Om_m = _omega_m(_EVAL_A, Om0, w0, wa)
        q = 1.0 - Om_m
        u = 1.0 + w0 + wa * (1.0 - _EVAL_A)
        
        # Features: [1, q, q^2, u, wa*a, u*q]
        phi = np.column_stack([
            np.ones_like(q),
            q,
            q ** 2,
            u,
            wa * _EVAL_A,
            u * q
        ])
        
        ln_Om = np.log(np.clip(Om_m, 1e-15, 1.0))
        ln_f = np.log(np.clip(f_r, 1e-15, 1.0))
        
        # Only use points where ln(Om) is well-defined (not too close to 1)
        mask = np.abs(ln_Om) > 0.01
        if np.sum(mask) > 10:
            rows.append(phi[mask])
            targets.append(ln_f[mask] / ln_Om[mask])
    
    A = np.vstack(rows)
    b = np.concatenate(targets)
    return A, b

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
        # Check gamma constraint
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

def _derivative_smoothed_objective(c, smooth_weight=0.01):
    """Objective with derivative penalty for smoothness (Chebyshev equioscillation)."""
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6
    
    base_err = _objective(c)
    
    # Add derivative smoothness penalty: penalize large changes in gamma
    max_der = 0.0
    for f_r, D_r, Om0, w0, wa in _REF:
        gam = _gamma(c, _EVAL_A, Om0, w0, wa)
        # Finite difference of gamma
        if len(_EVAL_A) > 1:
            dgam = np.diff(gam)
            max_der = max(max_der, np.max(np.abs(dgam)))
    
    return base_err + smooth_weight * max_der

# ── Optimization ────────────────────────────────────────────────────────────

def _lsq_logspace_fit():
    """Stage A: Linear least-squares fit in log-space for good initialization."""
    try:
        A, b = _build_lsq_matrix()
        if A is None or len(A) == 0:
            return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
        
        # Weighted least squares: weight by |ln(Om)| to emphasize regions where fit matters
        # Simple unweighted first
        c_lsq, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        
        # Clip to valid range
        c_lsq = np.clip(c_lsq, -10, 10)
        
        # Verify gamma constraint
        max_viol = 0.0
        for f_r, D_r, Om0, w0, wa in _REF:
            gam = _gamma(c_lsq, _EVAL_A, Om0, w0, wa)
            viol = np.maximum(0.0, 0.1 - gam) + np.maximum(0.0, gam - 1.2)
            max_viol = max(max_viol, np.max(viol))
        
        if max_viol > 0.05:
            # Too many violations, fall back to seed
            return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
        
        return c_lsq
    except Exception:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

def _optimize():
    # Stage A: Log-space linear pre-fit
    c_lsq = _lsq_logspace_fit()
    best_c = c_lsq.copy()
    best_err = _objective(best_c)
    
    # Also try the seed
    seed_c = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    seed_err = _objective(seed_c)
    if seed_err < best_err:
        best_c = seed_c.copy()
        best_err = seed_err
    
    if not _deadline_ok():
        return np.clip(best_c, -10, 10)
    
    # Stage B1: Nelder-Mead with derivative smoothing
    try:
        res = minimize(_derivative_smoothed_objective, best_c, method='Nelder-Mead',
                       options={'maxiter': 600, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if res.fun < best_err + 0.01:  # allow slight increase in base for smoothness
            base_fun = _objective(res.x)
            if base_fun < best_err:
                best_err = base_fun
                best_c = res.x.copy()
    except Exception:
        pass
    
    if not _deadline_ok():
        return np.clip(best_c, -10, 10)
    
    # Stage B2: Pure minimax Nelder-Mead
    try:
        res = minimize(_objective, best_c, method='Nelder-Mead',
                       options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if res.fun < best_err:
            best_err = res.fun
            best_c = res.x.copy()
    except Exception:
        pass
    
    if not _deadline_ok():
        return np.clip(best_c, -10, 10)
    
    # Stage B3: L-BFGS-B
    try:
        res = minimize(_objective, best_c, method='L-BFGS-B',
                       bounds=[(-5, 5)] * 6,
                       options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-9})
        if res.fun < best_err:
            best_err = res.fun
            best_c = res.x.copy()
    except Exception:
        pass
    
    if not _deadline_ok():
        return np.clip(best_c, -10, 10)
    
    # Stage B4: Multiple Nelder-Mead restarts from Chebyshev-perturbed points
    rng = np.random.default_rng(42)
    n_restarts = 12
    for i in range(n_restarts):
        if not _deadline_ok():
            break
        # Use Chebyshev-inspired perturbations
        scale = 0.05 + 0.05 * np.cos(np.pi * i / n_restarts)
        pert = rng.normal(0, scale, 6)
        x0 = np.clip(best_c + pert, -5, 5)
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_err:
                best_err = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    return np.clip(best_c, -10, 10)

_COEFFS = _optimize()

def construct():
    return list(_COEFFS)
