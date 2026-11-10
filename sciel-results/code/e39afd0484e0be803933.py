# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

_DEADLINE = time.time() + 625.0
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
        P = lpmv(am, k, mu)  # Condon-Shortley phase included
        if m == 0:
            Y[:, :, idx] = N * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * N * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * N * P[:, None] * np.sin(am * phi)[None, :]

    Y_flat = Y.reshape(_NTH * _NPHI, 2 * k + 1)
    w_flat = np.outer(w, np.full(_NPHI, dphi)).ravel()

    return Y_flat, w_flat


def _objective_and_grad(c, Y, w):
    """
    Compute R^4 = S4 / S2^2 and its gradient.
    Minimizing R is equivalent to minimizing R^4.
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
    
    # 1. Random dense
    for _ in range(50):
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
    for _ in range(50):
        v = np.zeros(n)
        idx = rng.choice(n, size=rng.integers(2, 8), replace=False)
        v[idx] = rng.standard_normal(len(idx))
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            inits.append(v / norm)
            
    return inits


def rudin_problem10() -> np.ndarray:
    Y, w = _build_harmonics()
    k = _K
    n = 2 * k + 1
    rng = np.random.default_rng(123)
    
    best_val = np.inf
    best_c = None
    
    # We minimize R^4 which is scale invariant.
    # Constraint: c.T @ c = 1
    
    cons = [{'type': 'eq', 'fun': lambda c: c.dot(c) - 1.0, 'jac': lambda c: 2.0 * c}]
    
    inits = _make_inits(rng, k)
    
    # Shuffle inits to ensure diversity in time
    rng.shuffle(inits)
    
    # Define fun and jac once to avoid recreation overhead
    def fun(c):
        val, _ = _objective_and_grad(c, Y, w)
        return val
        
    def jac(c):
        _, grad = _objective_and_grad(c, Y, w)
        return grad
        
    # Phase 1: Coarse search
    # Run all inits with moderate maxiter
    results = []
    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 30.0:
            break
            
        try:
            res = minimize(fun, x0, method='trust-constr', jac=jac,
                           constraints=cons,
                           options={'maxiter': 100, 'gtol': 1e-6, 'xtol': 1e-8})
            
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
                
            if np.isfinite(res.fun):
                results.append((res.fun, res.x.copy()))
        except Exception:
            pass

    # Phase 2: Refine the top candidates
    # Sort results by value
    results.sort(key=lambda x: x[0])
    
    # Take top 10
    top_candidates = results[:10]
    
    for val, c in top_candidates:
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Refine with tighter tolerance and more iterations
        try:
            res = minimize(fun, c, method='trust-constr', jac=jac,
                           constraints=cons,
                           options={'maxiter': 300, 'gtol': 1e-8, 'xtol': 1e-10})
            
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    # If we still have time, do a local refinement on the best solution
    if best_c is not None and time.time() < _DEADLINE - 5.0:
        # Use a few more random restarts around the best solution
        for _ in range(5):
            if time.time() > _DEADLINE - 5.0:
                break
            x0 = best_c + 0.1 * rng.standard_normal(n)
            x0 = x0 / np.linalg.norm(x0)
            
            try:
                res = minimize(fun, x0, method='trust-constr', jac=jac,
                               constraints=cons,
                               options={'maxiter': 300, 'gtol': 1e-9, 'xtol': 1e-11})
                if res.success or res.fun < best_val:
                    c_opt = res.x
                    c_opt = c_opt / np.linalg.norm(c_opt)
                    val_opt = res.fun
                    if np.isfinite(val_opt) and val_opt < best_val:
                        best_val = val_opt
                        best_c = c_opt.copy()
            except Exception:
                pass

    # Final validation and scaling
    if best_c is None:
        best_c = np.ones(n) / np.sqrt(n)
        
    out = best_c.astype(np.float64)
    nrm = np.linalg.norm(out)
    if nrm < 1e-12:
        out = np.ones(n)
    else:
        out = out / nrm
        
    # Scale out to ensure max|c| is in valid range
    max_abs = np.max(np.abs(out))
    if max_abs > 1e-3:
        out = out / max_abs
    else:
        out = out / max_abs if max_abs > 0 else np.ones(n)
        
    # Double check validity
    if out.shape != (21,):
        out = np.ones(21)
    if np.any(np.isnan(out)) or np.any(np.isinf(out)):
        out = np.ones(21)
        
    return out


# EVOLVE-BLOCK-END
