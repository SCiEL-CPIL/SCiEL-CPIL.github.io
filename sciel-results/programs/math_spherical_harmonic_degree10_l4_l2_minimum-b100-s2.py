# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

_DEADLINE = time.time() + 615.0
_K = 10
_NTH = 70
_NPHI = 130


def _build_harmonics():
    """Return Y (nphi*nth, 2k+1), weights (nphi*nth,) for quadrature."""
    k = _K
    mu, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    # Build Y: shape (NTH, NPHI, 2k+1) then flatten to (NTH*NPHI, 2k+1)
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

    # Flatten to (NTH*NPHI, 2k+1)
    Y_flat = Y.reshape(_NTH * _NPHI, 2 * k + 1)

    # Weights: outer product of GL weights and dphi, flattened
    w_flat = np.outer(w, np.full(_NPHI, dphi)).ravel()

    return Y_flat, w_flat


def _objective_and_grad(c, Y, w):
    """
    Compute ||f||_4 / ||f||_2 and its gradient w.r.t. c.
    Scale-invariant: we work with the ratio directly.
    
    R = ||f||_4 / ||f||_2
    ||f||_2 = sqrt(sum w_i f_i^2)
    ||f||_4 = (sum w_i f_i^4)^{1/4}
    
    We minimize R. Since R is scale-invariant, we can fix ||c||_2 = 1
    and minimize ||f||_4 (equivalently minimize R^4 = ||f||_4^4 / ||f||_2^4).
    
    Actually, let's minimize R directly with its gradient.
    R = A / B where A = ||f||_4 = (sum w f^4)^{1/4}, B = ||f||_2 = (sum w f^2)^{1/2}
    
    dR/dc = (1/B) dA/dc - (A/B^2) dB/dc
    
    dA/dc = (1/4) A^{-3} * sum w * 4 f^3 * df/dc = A^{-3} * sum w f^3 (Y^T (w * f^3))
    Wait: A = (S4)^{1/4} where S4 = sum w_i f_i^4
    dA/dc_j = (1/4) S4^{-3/4} * sum_i w_i * 4 f_i^3 * Y_ij = S4^{-3/4} * (Y^T (w * f^3))_j
    = A^{-3} * (Y^T (w * f^3))_j  since A = S4^{1/4}, A^{-3} = S4^{-3/4}
    
    dB/dc_j = (1/2) S2^{-1/2} * sum_i w_i * 2 f_i * Y_ij = S2^{-1/2} * (Y^T (w * f))_j
    = B^{-1} * (Y^T (w * f))_j  since B = S2^{1/2}, B^{-1} = S2^{-1/2}
    
    dR/dc_j = (1/B) * A^{-3} * (Y^T(w*f^3))_j - (A/B^2) * B^{-1} * (Y^T(w*f))_j
            = A^{-3}/B * (Y^T(w*f^3))_j - A/B^3 * (Y^T(w*f))_j
            = (1/B^3) * [ B^{-2} A^{-3} ... let me redo
    
    Let me use a cleaner formulation. Minimize R^4 = S4 / S2^2.
    d(S4/S2^2)/dc = (dS4 * S2^2 - S4 * 2 S2 * dS2) / S2^4
    = (dS4 * S2 - 2 S4 * dS2) / S2^3
    
    dS4/dc = 4 * Y^T (w * f^3)
    dS2/dc = 2 * Y^T (w * f)
    
    So d(R^4)/dc = [4 * Y^T(w*f^3) * S2 - 2 S4 * 2 * Y^T(w*f)] / S2^3
    = [4 * S2 * Y^T(w*f^3) - 4 * S4 * Y^T(w*f)] / S2^3
    = 4 * [S2 * Y^T(w*f^3) - S4 * Y^T(w*f)] / S2^3
    
    Since R^4 is monotone in R, minimizing R^4 is equivalent.
    """
    f = Y @ c  # (N,)
    f2 = f * f
    f3 = f2 * f
    f4 = f2 * f2
    
    S2 = np.dot(w, f2)
    S4 = np.dot(w, f4)
    
    if S2 < 1e-15:
        return 1e12, np.zeros_like(c)
    
    # R^4 = S4 / S2^2
    val = S4 / (S2 * S2)
    
    # Gradient of R^4 w.r.t. c
    Yt_wf3 = Y.T @ (w * f3)  # (2k+1,)
    Yt_wf = Y.T @ (w * f)    # (2k+1,)
    
    grad = 4.0 * (S2 * Yt_wf3 - S4 * Yt_wf) / (S2 * S2 * S2)
    
    return val, grad


def _ratio_val(c, Y, w):
    """Compute R = ||f||_4 / ||f||_2 for reporting."""
    f = Y @ c
    f2 = f * f
    S2 = np.dot(w, f2)
    S4 = np.dot(w, f2 * f2)
    if S2 < 1e-15:
        return np.inf
    return (S4 ** 0.25) / np.sqrt(S2)


def _make_inits(rng, k):
    """Diverse initial directions on the unit sphere."""
    inits = []
    n = 2 * k + 1
    # Random
    for _ in range(20):
        v = rng.standard_normal(n)
        inits.append(v / np.linalg.norm(v))
    # Single dominant m
    for m in range(-k, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        inits.append(v)
    # Paired cos/sin
    for m in range(1, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(v / np.linalg.norm(v))
    # Alternating signs
    v = np.ones(n)
    v[::2] = -1.0
    inits.append(v / np.linalg.norm(v))
    # Random sparse
    for _ in range(8):
        v = np.zeros(n)
        idx = rng.choice(n, size=rng.integers(2, 9), replace=False)
        v[idx] = rng.standard_normal(len(idx))
        if np.linalg.norm(v) > 1e-12:
            inits.append(v / np.linalg.norm(v))
    return inits


def rudin_problem10() -> np.ndarray:
    Y, w = _build_harmonics()
    k = _K
    n = 2 * k + 1
    rng = np.random.default_rng(42)
    inits = _make_inits(rng, k)

    best_c = inits[0].copy()
    best_val = np.inf

    # Phase 1: short L-BFGS-B from many diverse starts
    scored = []
    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 150.0:
            break

        def fun(c):
            val, grad = _objective_and_grad(c, Y, w)
            return val, grad

        try:
            res = minimize(fun, x0, method="L-BFGS-B", jac=True,
                           options={"maxiter": 100, "maxls": 30,
                                    "ftol": 1e-14, "gtol": 1e-10})
            c_opt = res.x
            val_opt = _ratio_val(c_opt, Y, w)
            if np.isfinite(val_opt) and val_opt < best_val:
                best_val = val_opt
                best_c = c_opt.copy()
            scored.append((val_opt, c_opt))
        except Exception:
            scored.append((np.inf, x0.copy()))

    # Sort and keep top candidates
    scored.sort(key=lambda t: t[0])
    top = [c for _, c in scored[:8] if np.isfinite(_)]
    if not top:
        top = [best_c]

    # Phase 2: refine top candidates with more iterations
    for x0 in top:
        if time.time() > _DEADLINE - 60.0:
            break

        def fun2(c):
            val, grad = _objective_and_grad(c, Y, w)
            return val, grad

        try:
            res = minimize(fun2, x0, method="L-BFGS-B", jac=True,
                           options={"maxiter": 500, "maxls": 40,
                                    "ftol": 1e-16, "gtol": 1e-12})
            c_opt = res.x
            val_opt = _ratio_val(c_opt, Y, w)
            if np.isfinite(val_opt) and val_opt < best_val:
                best_val = val_opt
                best_c = c_opt.copy()
        except Exception:
            pass

    # Phase 3: random restarts from perturbed best
    while time.time() < _DEADLINE - 30.0:
        x0 = best_c + 0.2 * rng.standard_normal(n)
        x0 = x0 / np.linalg.norm(x0)

        def fun3(c):
            val, grad = _objective_and_grad(c, Y, w)
            return val, grad

        try:
            res = minimize(fun3, x0, method="L-BFGS-B", jac=True,
                           options={"maxiter": 200, "maxls": 30,
                                    "ftol": 1e-15, "gtol": 1e-11})
            c_opt = res.x
            val_opt = _ratio_val(c_opt, Y, w)
            if np.isfinite(val_opt) and val_opt < best_val:
                best_val = val_opt
                best_c = c_opt.copy()
        except Exception:
            pass

    # Final: normalize and validate
    out = best_c.astype(np.float64)
    nrm = np.linalg.norm(out)
    if nrm < 1e-12:
        out = np.ones(n, dtype=np.float64)
    else:
        out = out / nrm

    # Validate
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


# EVOLVE-BLOCK-END
