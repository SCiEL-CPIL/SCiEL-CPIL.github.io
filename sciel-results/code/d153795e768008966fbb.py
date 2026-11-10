# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

# Precompute the real orthonormal spherical harmonics Y_m for degree 10, m=-10..10
def _build_harmonic_basis():
    k = 10
    n_theta = 70
    n_phi = 130

    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    m_vals = np.arange(0, 11)
    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = lpmv(m, 10, x)

    N = np.zeros(11)
    for m in range(11):
        num = 1.0
        for i in range(1, k - m + 1):
            num *= i
        den = 1.0
        for i in range(1, k + m + 1):
            den *= i
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) * (num / den))

    n_pts = n_theta * n_phi
    B = np.zeros((n_pts, 21))

    B[:, 10] = np.repeat(N[0] * P[0, :], n_phi)

    sqrt2 = np.sqrt(2.0)
    for m in range(1, 11):
        idx_pos = m + 10
        cos_term = np.cos(m * phi)
        val = sqrt2 * N[m] * P[m, :]
        B[:, idx_pos] = (val[:, None] * cos_term[None, :]).ravel()

        idx_neg = -m + 10
        sin_term = np.sin(m * phi)
        B[:, idx_neg] = (val[:, None] * sin_term[None, :]).ravel()

    weights_phi = np.full(n_phi, 2.0 * np.pi / n_phi)
    weights = np.repeat(w, n_phi)

    return B, weights, n_pts


B, weights, n_pts = _build_harmonic_basis()

def _compute_ratio(c):
    f = B @ c
    f2 = f * f
    f4 = f2 * f2
    norm2_sq = np.sum(weights * f2)
    norm4_4 = np.sum(weights * f4)

    if norm2_sq < 1e-30:
        return np.inf
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    return norm4 / norm2

def _objective_and_grad(c):
    """Compute the L4/L2 ratio and its gradient."""
    f = B @ c
    f2 = f * f
    f4 = f2 * f2

    norm2_sq = np.sum(weights * f2)
    norm4_4 = np.sum(weights * f4)

    if norm2_sq < 1e-30:
        return np.inf, np.zeros_like(c)

    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    ratio = norm4 / norm2

    # Gradient of ||f||_4 / ||f||_2
    grad_4 = 4.0 * (B.T @ (weights * f * f2))
    grad_2 = 2.0 * (B.T @ (weights * f))

    grad = (1.0 / (4.0 * norm4**3 * norm2)) * grad_4 - (norm4 / (2.0 * norm2**3)) * grad_2

    return ratio, grad

def _normalize_unit_l2(c):
    """Normalize c to have unit L2 norm on the sphere (using quadrature)."""
    f = B @ c
    norm2_sq = np.sum(weights * f * f)
    if norm2_sq < 1e-30:
        return c
    return c / np.sqrt(norm2_sq)

def _lbfgsb_refine(c0, deadline):
    """Refine with gradient-based L-BFGS-B, projecting to unit L2 each eval."""
    def obj_and_grad(c):
        if not np.all(np.isfinite(c)):
            return 1e10, np.zeros_like(c)
        cn = _normalize_unit_l2(c)
        val, grad = _objective_and_grad(cn)
        if not np.isfinite(val):
            return 1e10, np.zeros_like(c)
        return val, grad

    try:
        res = minimize(
            obj_and_grad,
            c0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 8000},
        )
        if res.fun < 1e10:
            cn = _normalize_unit_l2(res.x)
            return res.fun, cn
    except Exception:
        pass
    return np.inf, c0

def _nelder_mead_minimize(c0, deadline):
    """Nelder-Mead with deadline checking."""
    def objective(c):
        if time.time() > deadline:
            raise TimeoutError()
        r, _ = _objective_and_grad(c)
        return r

    try:
        result = minimize(
            objective,
            c0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
        )
        return result.x, result.fun
    except TimeoutError:
        return c0, _compute_ratio(c0)
    except Exception:
        return c0, _compute_ratio(c0)

def _power_iteration_start(deadline):
    """Power iteration on the 4th-order operator to find a low-R starting point."""
    dim = 21
    c = np.zeros(dim)
    c[10] = 1.0  # zonal
    c = _normalize_unit_l2(c)
    best_c = c.copy()
    best_val = _compute_ratio(c)

    for _ in range(200):
        if time.time() > deadline:
            break
        # Update direction: minimize ||f||_4^4 subject to ||f||_2^2=1
        # c_new ~ (B^T (w * f^3))  (gradient of ||f||_4^4)
        f = B @ c
        direction = B.T @ (weights * f**3)
        c_new = direction
        c_new = _normalize_unit_l2(c_new)
        val = _compute_ratio(c_new)
        if val < best_val:
            best_val = val
            best_c = c_new.copy()
        c = c_new
    return best_c, best_val

def _valid(c):
    return (len(c) == 21 and
            not np.any(np.isnan(c)) and
            not np.any(np.isinf(c)) and
            np.max(np.abs(c)) >= 1e-3 and
            np.max(np.abs(c)) <= 1e5)

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0  # safety margin before 630s kill

    dim = 21

    best_c = np.ones(dim, dtype=np.float64)
    best_val = _compute_ratio(best_c)

    # Strong seed: zonal harmonic (m=0) minimizes L4/L2 by maximizing spread
    zonal = np.zeros(dim)
    zonal[10] = 1.0
    zonal = _normalize_unit_l2(zonal)
    zv = _compute_ratio(zonal)
    if zv < best_val and _valid(zonal):
        best_val = zv
        best_c = zonal.copy()

    # Power iteration to refine the zonal basin
    if time.time() < deadline - 30.0:
        c_pi, val_pi = _power_iteration_start(deadline - 30.0)
        if val_pi < best_val and _valid(c_pi):
            best_val = val_pi
            best_c = c_pi.copy()

    # Generate diverse initializations
    np.random.seed(42)
    inits = []
    inits.append(best_c.copy())
    inits.append(zonal.copy())
    inits.append(np.ones(dim))
    for i in range(15):
        inits.append(np.random.randn(dim))
    for m_offset in range(dim):
        c_init = np.zeros(dim)
        c_init[m_offset] = 1.0
        inits.append(c_init)
    for m in range(1, 11):
        for sign in [1, -1]:
            c_init = np.zeros(dim)
            c_init[m + 10] = 1.0
            c_init[-m + 10] = sign
            inits.append(c_init)
    for n_nonzero in [2, 3, 5, 7]:
        for _ in range(4):
            c_init = np.zeros(dim)
            indices = np.random.choice(dim, size=n_nonzero, replace=False)
            c_init[indices] = np.random.choice([-1.0, 1.0], size=n_nonzero)
            inits.append(c_init)

    # Coarse Nelder-Mead pass from each initialization
    for i, c_init in enumerate(inits):
        if time.time() > deadline - 120.0:
            break
        c_norm = _normalize_unit_l2(c_init)
        c_opt, val = _nelder_mead_minimize(c_norm, deadline - 120.0)
        if val < best_val and _valid(c_opt):
            best_val = val
            best_c = c_opt.copy()

    # Gradient-based L-BFGS-B refinement from best and a few strong candidates
    refine_candidates = [best_c.copy(), zonal.copy()]
    if time.time() < deadline - 60.0:
        for rc in refine_candidates:
            if time.time() > deadline - 60.0:
                break
            rc_norm = _normalize_unit_l2(rc)
            c_opt, val = _lbfgsb_refine(rc_norm, deadline - 60.0)
            if val < best_val and _valid(c_opt):
                best_val = val
                best_c = c_opt.copy()
                refine_candidates.insert(0, best_c.copy())

    # Final L-BFGS-B refinement from best
    if time.time() < deadline - 30.0:
        c_norm = _normalize_unit_l2(best_c)
        c_opt, val = _lbfgsb_refine(c_norm, deadline - 30.0)
        if val < best_val and _valid(c_opt):
            best_val = val
            best_c = c_opt.copy()

    # Final validation
    c_final = best_c
    if len(c_final) != 21:
        c_final = np.ones(21)
    if np.any(np.isnan(c_final)) or np.any(np.isinf(c_final)):
        c_final = np.ones(21)
    max_abs = np.max(np.abs(c_final))
    if max_abs < 1e-3:
        c_final = c_final / max_abs if max_abs > 0 else np.ones(21)
    if np.max(np.abs(c_final)) > 1e5:
        c_final = c_final * (1e5 / np.max(np.abs(c_final)))

    return c_final

# EVOLVE-BLOCK-END
