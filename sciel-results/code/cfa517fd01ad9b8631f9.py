# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

_DEADLINE = time.time() + 610.0  # safety margin before 630s kill
_K = 10
_NTH = 70
_NPHI = 130


def _build_harmonics():
    """
    Build the real orthonormal degree-10 spherical harmonics on the
    70x130 quadrature grid. Returns (Y, w) where:
      Y has shape (NTH*NPHI, 21) and w has shape (NTH*NPHI,).
    Column index i corresponds to m = i - 10.
    """
    k = _K
    mu, w_theta = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    Y = np.zeros((_NTH, _NPHI, 2 * k + 1), dtype=np.float64)
    for idx, m in enumerate(range(-k, k + 1)):
        am = abs(m)
        N = np.sqrt((2 * k + 1) / (4.0 * np.pi)
                    * float(factorial(k - am, exact=True))
                    / float(factorial(k + am, exact=True)))
        P = lpmv(am, k, mu)  # Condon-Shortley phase included
        if m == 0:
            Y[:, :, idx] = N * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * N * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * N * P[:, None] * np.sin(am * phi)[None, :]

    Y_flat = Y.reshape(_NTH * _NPHI, 2 * k + 1)
    w_flat = np.outer(w_theta, np.full(_NPHI, dphi)).ravel()

    return Y_flat, w_flat


def _objective_and_grad(c, Y, w):
    """
    Compute R^4 = S4 / S2^2 and its gradient.
    Minimizing R is equivalent to minimizing R^4 (scale-invariant).
    """
    f = Y @ c
    f2 = f * f
    f3 = f2 * f
    f4 = f2 * f2

    S2 = np.dot(w, f2)
    S4 = np.dot(w, f4)

    if S2 < 1e-15:
        return 1e12, np.zeros_like(c)

    val = S4 / (S2 * S2)

    Yt_wf3 = Y.T @ (w * f3)
    Yt_wf = Y.T @ (w * f)

    grad = 4.0 * (S2 * Yt_wf3 - S4 * Yt_wf) / (S2 * S2 * S2)

    return val, grad


def _ratio_val(c, Y, w):
    """Compute the actual R = ||f||_4 / ||f||_2 for reporting/validation."""
    f = Y @ c
    f2 = f * f
    S2 = np.dot(w, f2)
    S4 = np.dot(w, f2 * f2)
    if S2 < 1e-15:
        return np.inf
    return (S4 ** 0.25) / np.sqrt(S2)


def _make_inits(rng, k):
    """Generate diverse initial directions on the unit sphere in R^(2k+1)."""
    inits = []
    n = 2 * k + 1

    # 1. Random dense
    for _ in range(40):
        v = rng.standard_normal(n)
        inits.append(v / np.linalg.norm(v))

    # 2. Basis vectors
    for m in range(-k, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        inits.append(v)

    # 3. Pairs (cos/sin)
    for m in range(1, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0 / np.sqrt(2.0)
        v[-m + k] = 1.0 / np.sqrt(2.0)
        inits.append(v)

        v2 = np.zeros(n)
        v2[m + k] = 1.0 / np.sqrt(2.0)
        v2[-m + k] = -1.0 / np.sqrt(2.0)
        inits.append(v2)

    # 4. Sparse random
    for _ in range(40):
        v = np.zeros(n)
        idx = rng.choice(n, size=rng.integers(2, 8), replace=False)
        v[idx] = rng.standard_normal(len(idx))
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            inits.append(v / norm)

    # 5. Structured: emphasize low |m| (flatter near poles)
    for _ in range(20):
        v = np.zeros(n)
        # Pick random subset of low |m|
        low_m = list(range(-3, 4))
        subset = rng.choice(low_m, size=rng.integers(2, 5), replace=False)
        for m in subset:
            v[m + k] = rng.standard_normal()
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            inits.append(v / norm)

    return inits


def _lbfgs_run(c0, Y, w, maxiter=300, ftol=1e-12, gtol=1e-10):
    """Run L-BFGS-B on R^4 objective. Returns (best_c, best_val)."""
    best_c = c0.copy()
    best_val = _objective_and_grad(c0, Y, w)[0]

    def fun(c):
        return _objective_and_grad(c, Y, w)[0]

    def jac(c):
        return _objective_and_grad(c, Y, w)[1]

    try:
        res = minimize(fun, c0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x
    except Exception:
        pass

    return best_c, best_val


def _trust_constr_run(c0, Y, w, maxiter=200, gtol=1e-8, xtol=1e-10):
    """Run trust-constr with unit-norm constraint on R^4 objective."""
    n = len(c0)
    best_c = c0.copy()
    best_val = _objective_and_grad(c0, Y, w)[0]

    cons = [{'type': 'eq', 'fun': lambda c: c.dot(c) - 1.0, 'jac': lambda c: 2.0 * c}]

    def fun(c):
        return _objective_and_grad(c, Y, w)[0]

    def jac(c):
        return _objective_and_grad(c, Y, w)[1]

    try:
        res = minimize(fun, c0, method='trust-constr', jac=jac,
                       constraints=cons,
                       options={'maxiter': maxiter, 'gtol': gtol, 'xtol': xtol})
        if res.fun < best_val:
            c_opt = res.x
            norm = np.linalg.norm(c_opt)
            if norm > 1e-12:
                c_opt = c_opt / norm
                best_val = res.fun
                best_c = c_opt
    except Exception:
        pass

    return best_c, best_val


def _rayleigh_polish(c, Y, w, n_iter=50, tol=1e-12):
    """
    Rayleigh quotient iteration: for the matrix formulation of R^4,
    iterate c <- (S4 * Y^T W Y - S2 * Y^T W Y diag(f^2) ... ) -- actually
    let's do a simpler steepest-descent step on the sphere.
    """
    c = c.copy()
    c = c / np.linalg.norm(c)
    for _ in range(n_iter):
        f = Y @ c
        f2 = f * f
        f3 = f2 * f
        S2 = np.dot(w, f2)
        S4 = np.dot(w, f2 * f2)
        if S2 < 1e-15:
            break
        # Gradient of R^4
        Yt_wf3 = Y.T @ (w * f3)
        Yt_wf = Y.T @ (w * f)
        grad = 4.0 * (S2 * Yt_wf3 - S4 * Yt_wf) / (S2 * S2 * S2)
        # Project to tangent space
        grad_tan = grad - np.dot(grad, c) * c
        # Adaptive step size
        step = 0.1
        for _ in range(20):
            c_new = c - step * grad_tan
            c_new = c_new / np.linalg.norm(c_new)
            new_val = _objective_and_grad(c_new, Y, w)[0]
            cur_val = S4 / (S2 * S2)
            if new_val < cur_val:
                break
            step *= 0.5
        else:
            break
        c = c_new
        if np.linalg.norm(grad_tan) < tol:
            break
    return c


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Returns np.ndarray of shape (21,) with coefficients c_m for m = -10..10.
    """
    Y, w = _build_harmonics()
    k = _K
    n = 2 * k + 1
    rng = np.random.default_rng(123)

    best_val = np.inf
    best_c = None

    inits = _make_inits(rng, k)
    rng.shuffle(inits)

    # Phase 1: L-BFGS-B from diverse inits
    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 120.0:
            break
        c_opt, val_opt = _lbfgs_run(x0, Y, w, maxiter=300, ftol=1e-12, gtol=1e-10)
        if np.isfinite(val_opt) and val_opt < best_val:
            best_val = val_opt
            best_c = c_opt.copy()

    # Phase 2: trust-constr from best and nearby points
    if best_c is not None and time.time() < _DEADLINE - 60.0:
        c_opt, val_opt = _trust_constr_run(best_c, Y, w, maxiter=200, gtol=1e-9, xtol=1e-11)
        if np.isfinite(val_opt) and val_opt < best_val:
            best_val = val_opt
            best_c = c_opt.copy()

    # Phase 3: Rayleigh polish
    if best_c is not None and time.time() < _DEADLINE - 30.0:
        c_opt = _rayleigh_polish(best_c, Y, w, n_iter=100, tol=1e-13)
        val_opt = _objective_and_grad(c_opt, Y, w)[0]
        if np.isfinite(val_opt) and val_opt < best_val:
            best_val = val_opt
            best_c = c_opt.copy()

    # Phase 4: More random restarts if time permits
    if best_c is not None and time.time() < _DEADLINE - 20.0:
        for _ in range(15):
            if time.time() > _DEADLINE - 15.0:
                break
            x0 = best_c + 0.05 * rng.standard_normal(n)
            x0 = x0 / np.linalg.norm(x0)
            c_opt, val_opt = _lbfgs_run(x0, Y, w, maxiter=200, ftol=1e-13, gtol=1e-11)
            if np.isfinite(val_opt) and val_opt < best_val:
                best_val = val_opt
                best_c = c_opt.copy()

    # Final polish
    if best_c is not None and time.time() < _DEADLINE - 5.0:
        c_opt = _rayleigh_polish(best_c, Y, w, n_iter=100, tol=1e-14)
        val_opt = _objective_and_grad(c_opt, Y, w)[0]
        if np.isfinite(val_opt) and val_opt < best_val:
            best_val = val_opt
            best_c = c_opt.copy()

    # Validation and scaling
    if best_c is None:
        best_c = np.ones(n) / np.sqrt(n)

    out = best_c.astype(np.float64)
    nrm = np.linalg.norm(out)
    if nrm < 1e-12:
        out = np.ones(n)
    else:
        out = out / nrm

    # Scale so max|c| = 1.0 (safely within [1e-3, 1e5])
    max_abs = np.max(np.abs(out))
    if max_abs > 1e-12:
        out = out / max_abs

    # Double check validity
    if out.shape != (21,):
        out = np.ones(21)
    if np.any(np.isnan(out)) or np.any(np.isinf(out)):
        out = np.ones(21)
    max_abs = np.max(np.abs(out))
    if max_abs < 1e-3 or max_abs > 1e5:
        out = np.ones(21)

    return out


# EVOLVE-BLOCK-END
