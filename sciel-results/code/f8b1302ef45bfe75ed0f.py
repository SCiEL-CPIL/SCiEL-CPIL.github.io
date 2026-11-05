"""Growth-index approximation for CPL dark-energy cosmologies.

Structural changes from parent:
1. Fast ODE reference: 200-point grid, loose tolerances to cut build time.
2. Log-space least-squares initialization for gamma coefficients.
3. Multi-stage optimization: Nelder-Mead, Powell, then adaptive multi-start.
4. RBF-smoothed gamma target for more stable initialization.
5. Deadline guard with 8s safety margin.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time
import warnings

warnings.filterwarnings("ignore")

_START_TIME = time.time()
_TIME_LIMIT = 82.0  # 8s margin for 90s limit


def _deadline_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT


# ── Physics helpers ──────────────────────────────────────────────────────────

def _omega_m_of_a(a, Om0, w0, wa):
    """Matter density parameter as a function of scale factor."""
    a = np.asarray(a, dtype=float)
    a_safe = np.clip(a, 1e-12, 1e6)
    a_inv3 = a_safe ** (-3.0)
    de_factor = a_safe ** (-3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (a_safe - 1.0))
    Om_DE = (1.0 - Om0) * de_factor
    denom = Om0 * a_inv3 + Om_DE
    Om_m = Om0 * a_inv3 / denom
    return Om_m


def _design_matrix(a, Om0, w0, wa):
    """Build the 6-column design matrix for gamma = X @ c."""
    a = np.asarray(a, dtype=float)
    Om_m = _omega_m_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    X = np.column_stack([
        np.ones_like(a),
        q,
        q ** 2,
        u,
        wa * a,
        u * q
    ])
    return X


# ── Reference: solve the growth ODE numerically (fast version) ──────────────

def _growth_ode_reference(a_grid, Om0, w0, wa):
    """
    Compute f(a) and D(a)/D(1) by solving the linear growth ODE for f.
    f' + (3 - 1.5*Om_m) f - 1.5*Om_m = 0
    Uses reduced grid and tolerances for speed.
    """
    a_min = a_grid[0]
    a_max = a_grid[-1]
    x_min = np.log(a_min)
    x_max = np.log(a_max)

    def ode_f(x, y):
        a = np.exp(x)
        Om_m = _omega_m_of_a(a, Om0, w0, wa)
        f = y[0]
        return [1.5 * Om_m - (3.0 - 1.5 * Om_m) * f]

    try:
        sol = solve_ivp(ode_f, [x_min, x_max], [1.0],
                        t_eval=a_grid,
                        max_step=0.05, rtol=1e-8, atol=1e-10)

        if not sol.success:
            raise RuntimeError("ODE integration failed")

        f_ref = sol.y[0]

        # Compute D_ref from f_ref via cumulative trapezoidal integration
        x_eval = np.log(a_grid)
        dx = np.diff(x_eval)
        f_avg = 0.5 * (f_ref[1:] + f_ref[:-1])
        cum_int = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
        total_int = cum_int[-1]
        J_vals = cum_int - total_int
        D_ref = np.exp(J_vals)
        D_ref = D_ref / D_ref[-1]

        return f_ref, D_ref

    except Exception:
        # Fallback: constant gamma approximation
        gamma_ref = 0.55
        Om_m_arr = _omega_m_of_a(a_grid, Om0, w0, wa)
        f_ref = Om_m_arr ** gamma_ref
        x = np.log(a_grid)
        dx = np.diff(x)
        f_avg = 0.5 * (f_ref[1:] + f_ref[:-1])
        cum_int = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
        integral_vals = cum_int - cum_int[-1]
        D_ref = np.exp(integral_vals)
        D_ref = D_ref / D_ref[-1]
        return f_ref, D_ref


# ── Validation cosmologies and evaluation grid ───────────────────────────────

_VALIDATION_COSMOLOGIES = [
    (0.30, -0.5, 0.0),
    (0.30, -1.0, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.30, -0.8, 0.3),
    (0.25, -1.0, 0.5),
    (0.35, -1.0, -0.3),
    (0.30, -1.2, 0.4),
    (0.30, -0.9, -0.4),
    (0.28, -1.0, 1.0),
    (0.22, -1.0, 0.8),
    (0.38, -1.0, -0.6),
    (0.30, -0.7, 0.6),
    (0.25, -1.2, 0.3),
    (0.35, -0.8, -0.5),
]

# Use a fine grid for evaluation
_EVAL_A = np.logspace(np.log10(1e-4), 0.0, 200)

_REF_DATA = []


def _build_references():
    """Pre-compute reference growth data for all validation cosmologies."""
    global _REF_DATA
    _REF_DATA = []
    for (Om0, w0, wa) in _VALIDATION_COSMOLOGIES:
        try:
            f_ref, D_ref = _growth_ode_reference(_EVAL_A, Om0, w0, wa)
            X = _design_matrix(_EVAL_A, Om0, w0, wa)
            Om_m = _omega_m_of_a(_EVAL_A, Om0, w0, wa)
            _REF_DATA.append((f_ref, D_ref, Om0, w0, wa, X, Om_m))
        except Exception:
            _REF_DATA.append(None)

    _REF_DATA = [r for r in _REF_DATA if r is not None]


def _compute_D_from_f(f_vals):
    """Compute normalized D from f values using trapezoidal integration."""
    x_eval = np.log(_EVAL_A)
    dx = np.diff(x_eval)
    f_avg = 0.5 * (f_vals[1:] + f_vals[:-1])
    cum_int = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
    J_vals = cum_int - cum_int[-1]
    D = np.exp(J_vals)
    return D / D[-1]


# ── Objective function ───────────────────────────────────────────────────────

def _objective(c):
    """Compute 100 * max relative error over all validation cosmologies."""
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6

    max_err = 0.0

    for ref in _REF_DATA:
        f_ref, D_ref, Om0, w0, wa, X, Om_m = ref

        gamma = X @ c

        # Check gamma constraint [0.1, 1.2]
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            violation = np.maximum(0.0, 0.1 - gamma) + np.maximum(0.0, gamma - 1.2)
            max_err = max(max_err, 10.0 * np.max(violation))
            continue

        Om_m_safe = np.clip(Om_m, 1e-15, 1.0)
        f_model = Om_m_safe ** gamma

        # Relative errors for f
        denom_f = np.maximum(np.abs(f_ref), 1e-10)
        rel_err_f = np.abs(f_model - f_ref) / denom_f

        # Compute D_model
        D_model = _compute_D_from_f(f_model)

        # Relative errors for D
        denom_D = np.maximum(np.abs(D_ref), 1e-10)
        rel_err_D = np.abs(D_model - D_ref) / denom_D

        max_err = max(max_err, np.max(rel_err_f), np.max(rel_err_D))

    return 100.0 * max_err


# ── Pre-compute references at module level ───────────────────────────────────

_build_references()


# ── Initialization strategies ────────────────────────────────────────────────

def _initial_guess_lsqr():
    """Least-squares fitting: log(f_ref) = gamma * log(Om_m)."""
    all_X = []
    all_target = []

    for ref in _REF_DATA:
        f_ref, D_ref, Om0, w0, wa, X, Om_m = ref
        Om_m_safe = np.clip(Om_m, 1e-15, 1.0)
        log_f = np.log(np.clip(f_ref, 1e-30, None))
        log_Om = np.log(Om_m_safe)
        with np.errstate(divide='ignore', invalid='ignore'):
            gamma_target = log_f / log_Om
        valid = np.isfinite(gamma_target)
        all_X.append(X[valid])
        all_target.append(gamma_target[valid])

    if len(all_X) == 0:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

    X_stack = np.vstack(all_X)
    target_stack = np.concatenate(all_target)

    try:
        result, _, _, _ = np.linalg.lstsq(X_stack, target_stack, rcond=None)
        return result
    except Exception:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])


def _initial_guess_rbf():
    """RBF-smoothed gamma target for more stable initialization.
    
    Uses radial basis function interpolation of the target gamma values
    to get a smoother, less noisy initial estimate.
    """
    all_X = []
    all_target = []
    all_a = []

    for ref in _REF_DATA:
        f_ref, D_ref, Om0, w0, wa, X, Om_m = ref
        Om_m_safe = np.clip(Om_m, 1e-15, 1.0)
        log_f = np.log(np.clip(f_ref, 1e-30, None))
        log_Om = np.log(Om_m_safe)
        with np.errstate(divide='ignore', invalid='ignore'):
            gamma_target = log_f / log_Om
        valid = np.isfinite(gamma_target)
        all_X.append(X[valid])
        all_target.append(gamma_target[valid])
        all_a.append(_EVAL_A[valid])

    if len(all_X) == 0:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

    X_stack = np.vstack(all_X)
    target_stack = np.concatenate(all_target)
    a_stack = np.concatenate(all_a)

    # Weight points by inverse distance to typical values for stability
    # Use a simple RBF kernel with Gaussian
    # Sample a few anchor points in the design space
    n_anchors = 20
    rng = np.random.default_rng(42)
    anchors = X_stack[rng.choice(len(X_stack), n_anchors, replace=False)]

    # Compute RBF weights
    sigma = np.median(np.linalg.norm(anchors[:, np.newaxis] - anchors[np.newaxis, :], axis=2)) * 0.5
    if sigma < 1e-10:
        sigma = 0.1

    # Build RBF interpolation matrix
    K = np.exp(-np.sum((anchors[:, np.newaxis] - anchors[np.newaxis, :]) ** 2, axis=2) / (2 * sigma ** 2))
    K += 1e-8 * np.eye(n_anchors)

    # Solve for RBF coefficients
    try:
        K_inv = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        return _initial_guess_lsqr()

    # Fit target at anchor points
    target_anchors = target_stack[rng.choice(len(target_stack), n_anchors, replace=False)]
    try:
        rbf_coeffs = K_inv @ target_anchors
    except Exception:
        return _initial_guess_lsqr()

    # Now fit gamma = X @ c using RBF-smoothed targets
    # For each point, compute RBF-interpolated gamma
    D_mat = np.exp(-np.sum((X_stack[:, np.newaxis] - anchors[np.newaxis, :]) ** 2, axis=2) / (2 * sigma ** 2))
    gamma_smoothed = D_mat @ rbf_coeffs

    # Clip to reasonable range
    gamma_smoothed = np.clip(gamma_smoothed, 0.1, 1.2)

    try:
        result, _, _, _ = np.linalg.lstsq(X_stack, gamma_smoothed, rcond=None)
        return result
    except Exception:
        return _initial_guess_lsqr()


def _run_optimization():
    """Run multi-stage optimization with deadline guard."""
    # Try both initialization strategies and pick the better one
    c_lsqr = _initial_guess_lsqr()
    c_lsqr = np.clip(c_lsqr, -10, 10)
    err_lsqr = _objective(c_lsqr)

    if not _deadline_ok():
        return c_lsqr

    c_rbf = _initial_guess_rbf()
    c_rbf = np.clip(c_rbf, -10, 10)
    err_rbf = _objective(c_rbf)

    if err_rbf < err_lsqr:
        best_c = c_rbf
        best_err = err_rbf
    else:
        best_c = c_lsqr
        best_err = err_lsqr

    if not _deadline_ok():
        return best_c

    # Stage 1: Nelder-Mead from best initial
    try:
        result = minimize(
            _objective, best_c, method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if result.fun < best_err:
            best_err = result.fun
            best_c = result.x.copy()
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage 2: Powell
    try:
        result = minimize(
            _objective, best_c, method='Powell',
            options={'maxiter': 2000, 'xtol': 1e-8, 'ftol': 1e-10}
        )
        if result.fun < best_err:
            best_err = result.fun
            best_c = result.x.copy()
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage 3: Multiple random restarts with Nelder-Mead
    rng = np.random.default_rng(123)
    n_restarts = 50
    for i in range(n_restarts):
        if not _deadline_ok():
            break
        scale = 0.2 * (0.5 ** (i // 20))
        x_try = best_c + rng.normal(0, scale, 6)
        x_try = np.clip(x_try, -5, 5)
        try:
            result = minimize(
                _objective, x_try, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_err:
                best_err = result.fun
                best_c = result.x.copy()
        except Exception:
            pass

    best_c = np.clip(best_c, -10, 10)
    return best_c


# Run optimization at import time
_COEFFS = _run_optimization()


def construct():
    """Return the six optimized growth-index coefficients."""
    return list(_COEFFS)
