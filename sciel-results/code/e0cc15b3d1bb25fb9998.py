import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 620.0
_K = 10
_NTH = 70
_NPHI = 130


def _build_harmonics():
    """Return Y (nphi*nth, 2k+1), weights (nphi*nth,) for quadrature."""
    k = _K
    mu, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    Y = np.zeros((_NTH, _NPHI, 2 * k + 1), dtype=np.float64)
    for idx, m in enumerate(range(-k, k + 1)):
        am = abs(m)
        N = np.sqrt((2 * k + 1) / (4.0 * np.pi)
                    * float(factorial(k - am, exact=True))
                    / float(factorial(k + am, exact=True)))
        P = lpmv(am, k, mu)
        if m == 0:
            Y[:, :, idx] = N * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * N * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * N * P[:, None] * np.sin(am * phi)[None, :]

    Y_flat = Y.reshape(_NTH * _NPHI, 2 * k + 1)
    w_flat = np.outer(w, np.full(_NPHI, dphi)).ravel()

    return Y_flat, w_flat


def _objective_ratio(c, Y, w):
    """Compute R = ||f||_4 / ||f||_2"""
    f = Y @ c
    f2 = f * f
    S2 = np.dot(w, f2)
    if S2 < 1e-15:
        return np.inf
    S4 = np.dot(w, f2 * f2)
    return (S4 ** 0.25) / np.sqrt(S2)


def _sphere_project(c, Y, w):
    """
    Compute R^4 and its gradient projected onto the tangent space of the unit sphere.
    Returns (val, grad_tangent) where grad_tangent is the projection of the gradient
    onto the tangent plane at c (i.e., grad - (grad . c) c).
    """
    f = Y @ c
    f2 = f * f
    f3 = f2 * f
    f4 = f2 * f2

    S2 = np.dot(w, f2)
    S4 = np.dot(w, f4)

    if S2 < 1e-15:
        return 1e12, np.zeros_like(c)

    # Minimize R^4 = S4 / S2^2
    val = S4 / (S2 * S2)

    # Gradient of R^4 with respect to c
    # d/d c [ S4 / S2^2 ] = (4 S2 dS4/dc - 2 S4 dS2/dc) / S2^4
    # dS4/dc = 4 Y^T (w f^3)
    # dS2/dc = 2 Y^T (w f)
    Yt_wf3 = Y.T @ (w * f3)
    Yt_wf = Y.T @ (w * f)

    grad_full = 4.0 * (S2 * Yt_wf3 - S4 * Yt_wf) / (S2 * S2 * S2)

    # Project onto tangent space: grad_t = grad_full - (grad_full . c) c
    dot = np.dot(grad_full, c)
    grad_t = grad_full - dot * c

    return val, grad_t


def _local_optimize(c0, Y, w, max_iter=500):
    """
    Trust-region / Newton-like method on the unit sphere.
    Uses R^4 as objective, gradient projected to tangent space.
    """
    c = c0.copy()
    n = len(c)
    norm = np.linalg.norm(c)
    if norm < 1e-12:
        return c0, np.inf
    c = c / norm

    best_val, best_c = np.inf, c.copy()

    for it in range(max_iter):
        val, grad = _sphere_project(c, Y, w)

        if not np.isfinite(val):
            break

        if val < best_val:
            best_val = val
            best_c = c.copy()

        # Check convergence
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-14:
            break

        # Newton step: Hessian approximation via finite differences or analytical
        # For now, use steepest descent with adaptive step size (more robust)
        # Actually, let's do a proper line search on the sphere

        # Direction: -grad (already in tangent space)
        d = -grad / (gnorm + 1e-30)

        # Line search: minimize val(c + alpha * d) projected to sphere
        # Use backtracking
        alpha = 1.0
        c_trial = c + alpha * d
        nrm = np.linalg.norm(c_trial)
        if nrm < 1e-12:
            break
        c_trial = c_trial / nrm

        val_trial, _ = _sphere_project(c_trial, Y, w)

        # Backtracking
        while val_trial > val - 1e-4 * alpha * gnorm * gnorm and alpha > 1e-10:
            alpha *= 0.5
            c_trial = c + alpha * d
            nrm = np.linalg.norm(c_trial)
            if nrm < 1e-12:
                break
            c_trial = c_trial / nrm
            val_trial, _ = _sphere_project(c_trial, Y, w)

        c = c_trial

    return best_c, best_val


def _unconstrained_objective(x, Y, w):
    """
    Objective for global optimizers working in [-1, 1] bounds.
    Maps x to unit sphere.
    """
    c = x
    norm = np.linalg.norm(c)
    if norm < 1e-12:
        return 1e10
    c_norm = c / norm
    f = Y @ c_norm
    f2 = f * f
    S2 = np.dot(w, f2)
    if S2 < 1e-15:
        return 1e10
    S4 = np.dot(w, f2 * f2)
    return S4 / (S2 * S2)


def rudin_problem10() -> np.ndarray:
    Y, w = _build_harmonics()
    k = _K
    n = 2 * k + 1

    best_c = np.zeros(n)
    best_val = np.inf

    # 1. Generate diverse initial guesses
    inits = []
    # m=0 only
    v0 = np.zeros(n); v0[k] = 1.0; inits.append(v0)
    # m=10 only (cos)
    v10 = np.zeros(n); v10[2*k] = 1.0; inits.append(v10)
    # m=10 only (sin)
    v10s = np.zeros(n); v10s[k-10] = 1.0; inits.append(v10s)
    # Uniform
    v_u = np.ones(n); inits.append(v_u / np.linalg.norm(v_u))
    # Random
    rng = np.random.default_rng(123)
    for _ in range(30):
        v = rng.standard_normal(n)
        inits.append(v / np.linalg.norm(v))

    # 2. Local Optimization from Inits using sphere-projected gradient descent
    for x0 in inits:
        if time.time() > _DEADLINE - 150.0:
            break
        try:
            c_opt, val_opt = _local_optimize(x0, Y, w, max_iter=300)
            val_r = _objective_ratio(c_opt, Y, w)
            if val_r < best_val and np.isfinite(val_r):
                best_val = val_r
                best_c = c_opt.copy()
        except Exception:
            pass

    # 3. Global Optimization: Differential Evolution
    bounds = [(-1.0, 1.0)] * n
    de_obj = lambda x: _unconstrained_objective(x, Y, w)

    try:
        if time.time() < _DEADLINE - 180.0:
            res_de = differential_evolution(
                de_obj, bounds, seed=42, maxiter=1000, popsize=20,
                tol=1e-8, mutation=(0.3, 1.0), recombination=0.9,
                polish=False
            )
            c_de = res_de.x
            norm = np.linalg.norm(c_de)
            if norm > 1e-12:
                c_de_norm = c_de / norm
                # Polish with local method
                c_polished, val_polished = _local_optimize(c_de_norm, Y, w, max_iter=300)
                val_de = _objective_ratio(c_polished, Y, w)
                if val_de < best_val and np.isfinite(val_de):
                    best_val = val_de
                    best_c = c_polished.copy()
    except Exception:
        pass

    # 4. Final Polish with tighter tolerance
    if time.time() < _DEADLINE - 20.0:
        try:
            c_final, val_final = _local_optimize(best_c, Y, w, max_iter=500)
            val_final_r = _objective_ratio(c_final, Y, w)
            if val_final_r < best_val and np.isfinite(val_final_r):
                best_val = val_final_r
                best_c = c_final.copy()
        except Exception:
            pass

    # Normalize and validate
    out = best_c.astype(np.float64)
    nrm = np.linalg.norm(out)
    if nrm < 1e-12:
        out = np.ones(n, dtype=np.float64)
    else:
        out = out / nrm

    if out.shape != (21,):
        out = np.ones(21, dtype=np.float64)
    if np.any(np.isnan(out)) or np.any(np.isinf(out)):
        out = np.ones(21, dtype=np.float64)

    max_abs = np.max(np.abs(out))
    if max_abs < 1e-3:
        out = out / max_abs if max_abs > 0 else np.ones(21)
    if max_abs > 1e5:
        out = out / max_abs

    return out
