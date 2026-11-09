"""Structural redesign: fit the six-coefficient compact growth-index ansatz
against an independent GR growth ODE reference, using a smooth surrogate
objective and multi-phase optimization with a hard wall-clock deadline guard.

The six coefficients returned by construct() are the *only* interface.
The optimization runs once inside construct() with phases:
  1. Pre-solve reference ODEs for all cosmologies (cached).
  2. Coarse search with a smooth (log-sum-exp) surrogate objective.
  3. Fine polish with the true max objective at high precision.
"""
import math
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Hard wall-clock guard
# ---------------------------------------------------------------------------
_DEADLINE = time.time() + 84.0
_MAX_COEFF = 10.0


def _remaining():
    return _DEADLINE - time.time()


# ---------------------------------------------------------------------------
# Cosmology helpers
# ---------------------------------------------------------------------------
def _Om(a, Om0, w0, wa):
    """Total matter density parameter at scale factor a (normalized to a=1)."""
    if a <= 0:
        return 1.0
    y = a ** (-3.0)
    # For CPL: rho_DE ~ a^(-3(1+w0)) * exp(3*wa*(1-a))
    # So H2(a)/H0^2 = Om0*a^-3 + (1-Om0)*a^(-3(1+w0)) * exp(3*wa*(1-a))
    de_exponent = -3.0 * (1.0 + w0) + 3.0 * wa * (1.0 - a)
    de = (1.0 - Om0) * math.exp(de_exponent) if (1.0 - Om0) > 0 else 0.0
    total = Om0 * y + de
    if total <= 0:
        return 1.0
    return Om0 * y / total


def _Om_grid(a_grid, Om0, w0, wa):
    """Vectorized Om for an array of a values."""
    a_grid = np.asarray(a_grid, dtype=float)
    y = a_grid ** (-3.0)
    de_exponent = -3.0 * (1.0 + w0) + 3.0 * wa * (1.0 - a_grid)
    de = np.where((1.0 - Om0) > 0, (1.0 - Om0) * np.exp(de_exponent), 0.0)
    total = Om0 * y + de
    total = np.maximum(total, 1e-30)
    return Om0 * y / total


def _gamma_of_a(a, Om0, w0, wa, c):
    q = 1.0 - _Om(a, Om0, w0, wa)
    u = 1.0 + w0 + wa * (1.0 - a)
    return (c[0] + c[1] * q + c[2] * q * q + c[3] * u
            + c[4] * wa * a + c[5] * u * q)


def _gamma_grid(a_grid, Om0, w0, wa, c):
    q = 1.0 - _Om_grid(a_grid, Om0, w0, wa)
    u = 1.0 + w0 + wa * (1.0 - a_grid)
    return (c[0] + c[1] * q + c[2] * q * q + c[3] * u
            + c[4] * wa * a_grid + c[5] * u * q)


# ---------------------------------------------------------------------------
# Reference: GR growth ODE in x = ln(a)
# D'' + (3/2 + 1/2 * d(ln H)/dx) D' - (3/2) Om D = 0
# ---------------------------------------------------------------------------
def _H2(a, Om0, w0, wa):
    """H(a)^2 / H0^2 for CPL dark energy."""
    if a <= 0:
        return 0.0
    y = a ** (-3.0)
    de_exponent = -3.0 * (1.0 + w0) + 3.0 * wa * (1.0 - a)
    de = (1.0 - Om0) * math.exp(de_exponent) if (1.0 - Om0) > 0 else 0.0
    return Om0 * y + de


def _rhs(ln_a, y, Om0, w0, wa):
    a = math.exp(ln_a)
    H2 = _H2(a, Om0, w0, wa)
    if H2 <= 0:
        return [y[1], 0.0]
    # d(ln H)/dx = (1/2) * d(ln H^2)/dx = (1/2) * (dH2/da) * (a / H2)
    # dH2/da = Om0*(-3)*a^(-4) + (1-Om0)*a^(-3(1+w0))*exp(3*wa*(1-a)) * d/da[-3(1+w0)+3*wa*(1-a)]
    #       = -3*Om0*a^(-4) + de * (-3*wa)
    de_exponent = -3.0 * (1.0 + w0) + 3.0 * wa * (1.0 - a)
    de = (1.0 - Om0) * math.exp(de_exponent) if (1.0 - Om0) > 0 else 0.0
    dH2_da = -3.0 * Om0 * a ** (-4.0) + de * (-3.0 * wa)
    dlnH_dx = 0.5 * dH2_da * a / H2
    Om = _Om(a, Om0, w0, wa)
    d1 = (1.5 + 0.5 * dlnH_dx) * y[1] - 1.5 * Om * y[0]
    return [y[1], d1]


def _solve_reference(a_grid, Om0, w0, wa):
    """Return (D, f) arrays on a_grid from the GR ODE, normalized D(1)=1."""
    ln_a = np.log(a_grid)
    # Integrate from a=1 backward to a_grid[0]
    sol = solve_ivp(_rhs, (ln_a[-1], ln_a[0]), [1.0, 0.0],
                    args=(Om0, w0, wa), method="RK45",
                    rtol=1e-9, atol=1e-12, dense_output=False,
                    t_eval=ln_a[::-1])
    if not sol.success:
        return None
    # sol.y[0] is D, sol.y[1] is D' = dD/dln(a)
    # They are in reverse order (from a=1 to a=a_grid[0])
    D = sol.y[0][::-1]
    Dp = sol.y[1][::-1]
    D = D / D[-1]                       # normalize D(1)=1
    f = Dp / D                          # d ln D / d ln a
    return D, f


# ---------------------------------------------------------------------------
# Candidate evaluation on a grid for one cosmology
# ---------------------------------------------------------------------------
def _eval_candidate(a_grid, Om0, w0, wa, c, D_ref, f_ref):
    Om = _Om_grid(a_grid, Om0, w0, wa)
    g = _gamma_grid(a_grid, Om0, w0, wa, c)
    # Clip gamma to valid range to avoid numerical issues
    g = np.clip(g, 0.1, 1.2)
    f_cand = Om ** g
    ln_a = np.log(a_grid)
    # D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
    # For a < 1: integral from ln(a) to 0 is positive, so D < 1
    # Compute cumulative integral from the right (a=1) to the left
    integrand = f_cand
    cum = np.zeros_like(ln_a)
    # cum[i] = integral from ln_a[i] to ln_a[-1]=0 of f dx
    for i in range(len(ln_a) - 2, -1, -1):
        x0 = ln_a[i]
        x1 = ln_a[i + 1]
        cum[i] = cum[i + 1] + 0.5 * (integrand[i] + integrand[i + 1]) * (x1 - x0)
    D_cand = np.exp(-cum)
    # Relative errors
    rel_f = np.abs(f_cand - f_ref) / np.maximum(np.abs(f_ref), 1e-15)
    rel_D = np.abs(D_cand - D_ref) / np.maximum(np.abs(D_ref), 1e-15)
    return float(np.max(np.maximum(rel_f, rel_D)))


def _eval_candidate_smooth(a_grid, Om0, w0, wa, c, D_ref, f_ref, beta=50.0):
    """Smooth surrogate using log-sum-exp for the max."""
    Om = _Om_grid(a_grid, Om0, w0, wa)
    g = _gamma_grid(a_grid, Om0, w0, wa, c)
    g = np.clip(g, 0.1, 1.2)
    f_cand = Om ** g
    ln_a = np.log(a_grid)
    integrand = f_cand
    cum = np.zeros_like(ln_a)
    for i in range(len(ln_a) - 2, -1, -1):
        x0 = ln_a[i]
        x1 = ln_a[i + 1]
        cum[i] = cum[i + 1] + 0.5 * (integrand[i] + integrand[i + 1]) * (x1 - x0)
    D_cand = np.exp(-cum)
    rel_f = np.abs(f_cand - f_ref) / np.maximum(np.abs(f_ref), 1e-15)
    rel_D = np.abs(D_cand - D_ref) / np.maximum(np.abs(D_ref), 1e-15)
    all_errs = np.concatenate([rel_f, rel_D])
    # Log-sum-exp for smooth max
    max_err = np.max(all_errs)
    if max_err < 1e-15:
        return 0.0
    shifted = all_errs / max_err
    logsumexp = np.log(np.sum(np.exp(beta * (shifted - 1.0)))) / beta + np.log(max_err)
    # This approximates (1/beta)*log(sum(exp(beta*err))) which is ~max for large beta
    smooth_max = max_err * (1.0 + np.log(np.mean(np.exp(beta * (all_errs / max_err - 1.0)))))
    return float(smooth_max)


# ---------------------------------------------------------------------------
# Objective: worst-case over cosmologies and grid
# ---------------------------------------------------------------------------
_COSMO = [(0.3, -1.0, 0.0), (0.3, -1.0, 0.3), (0.3, -1.0, -0.5),
          (0.27, -1.0, 0.0), (0.3, -0.9, 0.2), (0.3, -1.1, 0.4),
          (0.35, -1.0, 0.0), (0.25, -1.0, 0.5)]
_A_GRID = np.linspace(0.01, 1.0, 200)

# Cache for reference solutions
_ref_cache = {}


def _get_reference(Om0, w0, wa):
    key = (Om0, w0, wa)
    if key not in _ref_cache:
        _ref_cache[key] = _solve_reference(_A_GRID, Om0, w0, wa)
    return _ref_cache[key]


def _objective(c):
    c = np.clip(c, -_MAX_COEFF, _MAX_COEFF)
    worst = 0.0
    for (Om0, w0, wa) in _COSMO:
        ref = _get_reference(Om0, w0, wa)
        if ref is None:
            continue
        D_ref, f_ref = ref
        e = _eval_candidate(_A_GRID, Om0, w0, wa, c, D_ref, f_ref)
        if e > worst:
            worst = e
    return worst


def _objective_smooth(c, beta=30.0):
    c = np.clip(c, -_MAX_COEFF, _MAX_COEFF)
    worst = 0.0
    for (Om0, w0, wa) in _COSMO:
        ref = _get_reference(Om0, w0, wa)
        if ref is None:
            continue
        D_ref, f_ref = ref
        e = _eval_candidate_smooth(_A_GRID, Om0, w0, wa, c, D_ref, f_ref, beta=beta)
        if e > worst:
            worst = e
    return worst


# ---------------------------------------------------------------------------
# construct()
# ---------------------------------------------------------------------------
def construct():
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_val = float("inf")

    try:
        # Pre-solve references once
        for (Om0, w0, wa) in _COSMO:
            _get_reference(Om0, w0, wa)

        # Phase 1: Coarse search with smooth objective
        starts = [
            np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.5, 0.1, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.6, -0.1, 0.05, 0.0, 0.0, 0.0]),
            np.array([0.55, 0.0, 0.0, 0.05, 0.0, 0.0]),
            np.array([0.5, 0.05, 0.0, 0.05, 0.0, 0.0]),
            np.array([0.55, 0.05, -0.05, 0.0, 0.0, 0.0]),
            np.array([0.5, -0.05, 0.0, 0.02, 0.0, 0.0]),
        ]

        # Phase 1: Use smooth objective for faster convergence
        for s in starts:
            if _remaining() < 20.0:
                break
            try:
                res = minimize(_objective_smooth, s, method="Nelder-Mead",
                               options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-5,
                                        "adaptive": True})
                if res.fun < best_val:
                    cand = np.clip(res.x, -_MAX_COEFF, _MAX_COEFF)
                    v = _objective(cand)  # Evaluate with true objective
                    if v < best_val:
                        best_val = v
                        best_c = cand
            except Exception:
                pass

        # Phase 2: Fine polish with true objective
        if _remaining() > 15.0:
            polish_starts = [best_c]
            # Add some perturbations around the best found
            for i in range(5):
                if _remaining() < 10.0:
                    break
                perturbation = np.random.randn(6) * 0.02
                polish_starts.append(best_c + perturbation)

            for s in polish_starts:
                if _remaining() < 8.0:
                    break
                try:
                    res = minimize(_objective, s, method="Nelder-Mead",
                                   options={"maxiter": 300, "xatol": 1e-6, "fatol": 1e-7,
                                            "adaptive": True})
                    if res.fun < best_val:
                        cand = np.clip(res.x, -_MAX_COEFF, _MAX_COEFF)
                        v = _objective(cand)
                        if v < best_val:
                            best_val = v
                            best_c = cand
                except Exception:
                    pass

    except Exception:
        pass

    # Clamp to the required finite range
    out = [float(min(max(x, -_MAX_COEFF), _MAX_COEFF)) for x in best_c]
    return out
