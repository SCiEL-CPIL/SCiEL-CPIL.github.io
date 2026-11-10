# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial

_DEADLINE = time.time() + 610.0  # safety margin under the 630s kill
_K = 10
_NTH = 70
_NPHI = 130
_DIM = 2 * _K + 1  # 21


def _build_harmonics():
    """Return Y_flat (N, 2k+1), w_flat (N,) where N = NTH*NPHI, and dphi.
    
    Y_flat[i, j] = Y_j(theta_i, phi_j) for quadrature point i.
    w_flat[i] = w_theta(i) for the Gauss-Legendre weight.
    The measure element is w_flat[i] * dphi.
    """
    k = _K
    mu, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    # mu shape (NTH,), phi shape (NPHI,)
    # We want Y_flat of shape (NTH*NPHI, 2k+1)
    # For each (theta_idx, phi_idx), compute all 21 harmonics.
    Y_flat = np.zeros((_NTH * _NPHI, _DIM), dtype=np.float64)
    
    # Precompute cos(m*phi) and sin(m*phi) for m=0..k
    # cos_table[m, j] = cos(m * phi_j), sin_table[m, j] = sin(m * phi_j)
    cos_table = np.zeros((k + 1, _NPHI))
    sin_table = np.zeros((k + 1, _NPHI))
    for m in range(k + 1):
        cos_table[m] = np.cos(m * phi)
        sin_table[m] = np.sin(m * phi)

    # Precompute N_m and P^m(mu) for m=0..k
    N_m = np.zeros(k + 1)
    P_m = np.zeros((k + 1, _NTH))  # P_m[m_idx, th_idx] = P_k^m(mu[th_idx])
    for am in range(k + 1):
        N_m[am] = np.sqrt((2 * k + 1) / (4.0 * np.pi)
                          * float(factorial(k - am, exact=True))
                          / float(factorial(k + am, exact=True)))
        P_m[am] = lpmv(am, k, mu)

    # Fill Y_flat
    # Index in Y_flat: i = th_idx * NPHI + ph_idx
    for th_idx in range(_NTH):
        for ph_idx in range(_NPHI):
            i = th_idx * _NPHI + ph_idx
            for m in range(-k, k + 1):
                idx = m + k
                am = abs(m)
                if m == 0:
                    Y_flat[i, idx] = N_m[0] * P_m[0, th_idx]
                elif m > 0:
                    Y_flat[i, idx] = np.sqrt(2.0) * N_m[m] * P_m[m, th_idx] * cos_table[m, ph_idx]
                else:
                    Y_flat[i, idx] = np.sqrt(2.0) * N_m[am] * P_m[am, th_idx] * sin_table[am, ph_idx]

    # w_flat: each theta weight repeated NPHI times
    w_flat = np.tile(w, _NPHI)
    return Y_flat, w_flat, dphi


def _norms(Y_flat, w_flat, dphi, c):
    """Return (||f||_2, ||f||_4) for f = Y_flat @ c."""
    f = Y_flat @ c
    f2 = f * f
    f4 = f2 * f2
    n2 = np.sum(f2 * w_flat) * dphi
    n4 = np.sum(f4 * w_flat) * dphi
    return n2, n4


def _ratio_and_grad(Y_flat, w_flat, dphi, c):
    """Return (R, grad_R) where R = ||f||_4 / ||f||_2 and grad_R is the gradient w.r.t. c.
    
    We minimize R^4 = ||f||_4^4 / ||f||_2^4 to avoid roots, but the gradient of R
    is what we need. Actually, let's minimize R directly or R^4.
    R = (n4)^(1/4) / (n2)^(1/2)
    Let's compute gradient of R.
    
    Alternatively, minimize R^4 = n4 / n2^2. This is smoother.
    Let Q = n4 / n2^2. Minimize Q.
    grad Q = (grad n4 * n2^2 - n4 * 2 * n2 * grad n2) / n2^4
           = (grad n4 * n2 - 2 * n4/n2 * grad n2) / n2^3 ... 
    Let's just use the gradient of R directly with scipy's minimize, or use a trust-region method.
    
    Actually, the task suggests a trust-region method. Let's implement a simple trust-region
    algorithm on the sphere. But first, let's get the gradient of R.
    
    f = Y c
    n2 = sum(f^2 w) dphi
    n4 = sum(f^4 w) dphi
    R = n4^(1/4) / n2^(1/2)
    
    dR/dc = dR/dn4 * dn4/dc + dR/dn2 * dn2/dc
    dR/dn4 = (1/4) n4^(-3/4) / n2^(1/2)
    dR/dn2 = -n4^(1/4) * (1/2) n2^(-3/2)
    
    dn4/dc = d/dc sum(f^4 w) dphi = sum(4 f^3 (Y^T w) dphi) = 4 dphi Y^T (f^3 w)
    dn2/dc = d/dc sum(f^2 w) dphi = sum(2 f (Y^T w) dphi) = 2 dphi Y^T (f w)
    
    So grad R = (1/4) n4^(-3/4) n2^(-1/2) * 4 dphi Y^T (f^3 w)
               - n4^(1/4) (1/2) n2^(-3/2) * 2 dphi Y^T (f w)
             = dphi [ n4^(-3/4) n2^(-1/2) Y^T (f^3 w) - n4^(1/4) n2^(-3/2) Y^T (f w) ]
    
    Let's simplify:
    grad R = dphi * n4^(-3/4) * n2^(-3/2) [ n2 * Y^T (f^3 w) - n4 * Y^T (f w) ]
    """
    f = Y_flat @ c
    f2 = f * f
    f3 = f2 * f
    f4 = f2 * f2
    n2 = np.sum(f2 * w_flat) * dphi
    n4 = np.sum(f4 * w_flat) * dphi
    if n2 < 1e-12:
        return np.inf, np.zeros_like(c)
    R = (n4 ** 0.25) / np.sqrt(n2)
    
    # grad R
    term1 = Y_flat.T @ (f3 * w_flat)
    term2 = Y_flat.T @ (f * w_flat)
    grad_R = dphi * (n4 ** (-0.75)) * (n2 ** (-1.5)) * (n2 * term1 - n4 * term2)
    return R, grad_R


def _project_to_sphere(c):
    nrm = np.linalg.norm(c)
    if nrm < 1e-12:
        return c
    return c / nrm


def _trust_region_sphere(Y_flat, w_flat, dphi, x0, maxiter=500, delta0=0.5, eta=0.1, delta_max=5.0, delta_min=1e-6):
    """Trust-region method on the unit sphere to minimize R(c).
    
    At each step:
    1. Compute R(x), grad R(x)
    2. Project gradient to tangent space: g_tan = grad - (grad·x) x
    3. Solve trust-region subproblem: min g_tan^T d + 0.5 d^T H d s.t. ||d|| <= delta, d^T x = 0
       Approximate Hessian or use simple gradient step.
    4. For simplicity, use a damped gradient step in the tangent space with line search.
       Or use a proper trust-region step.
    
    Let's use a simpler approach: gradient descent with backtracking line search on the sphere.
    """
    x = _project_to_sphere(x0.copy())
    best_x = x.copy()
    best_R = _ratio_and_grad(Y_flat, w_flat, dphi, x)[0]
    delta = delta0
    
    for it in range(maxiter):
        if time.time() > _DEADLINE - 10:
            break
        R, g = _ratio_and_grad(Y_flat, w_flat, dphi, x)
        if R < best_R:
            best_R = R
            best_x = x.copy()
        if np.isinf(R):
            break
        
        # Project gradient to tangent space
        g_tan = g - np.dot(g, x) * x
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-10:
            break
        
        # Trust-region step: d = - (delta / g_norm) * g_tan if g_norm > delta, else -g_tan
        # This is a simple dogleg-like step.
        if g_norm > delta:
            d = -(delta / g_norm) * g_tan
        else:
            d = -g_tan
        
        # Move and project back to sphere
        x_new = x + d
        x_new = _project_to_sphere(x_new)
        R_new = _ratio_and_grad(Y_flat, w_flat, dphi, x_new)[0]
        
        # Check improvement
        if R_new < R - eta * np.dot(g_tan, d):
            # Accept
            x = x_new
            # Increase trust region
            delta = min(2.0 * delta, delta_max)
        else:
            # Reject
            delta = 0.5 * delta
            if delta < delta_min:
                break
    
    return best_x, best_R


def _make_inits(rng, k):
    """Diverse initial directions."""
    inits = []
    # random
    for _ in range(30):
        v = rng.standard_normal(_DIM)
        inits.append(v / np.linalg.norm(v))
    # single dominant m
    for m in range(-k, k + 1):
        v = np.zeros(_DIM)
        v[m + k] = 1.0
        inits.append(v)
    # paired cos/sin (m>0)
    for m in range(1, k + 1):
        v = np.zeros(_DIM)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(v / np.linalg.norm(v))
    # alternating signs
    v = np.ones(_DIM)
    v[::2] = -1.0
    inits.append(v / np.linalg.norm(v))
    # random sparse
    for _ in range(10):
        v = np.zeros(_DIM)
        idx = rng.choice(_DIM, size=rng.integers(2, 8), replace=False)
        v[idx] = rng.standard_normal(len(idx))
        if np.linalg.norm(v) > 0:
            inits.append(v / np.linalg.norm(v))
    # concentrated on high |m|
    for m in [k, k-1, k-2]:
        v = np.zeros(_DIM)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(v / np.linalg.norm(v))
    return inits


def rudin_problem10() -> np.ndarray:
    Y_flat, w_flat, dphi = _build_harmonics()
    k = _K
    rng = np.random.default_rng(42)
    inits = _make_inits(rng, k)

    best_c = inits[0].copy()
    best_val = np.inf

    # Phase 1: trust-region from diverse starts, keep top candidates
    scored = []
    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 120:
            break
        c, val = _trust_region_sphere(Y_flat, w_flat, dphi, x0, maxiter=300, delta0=0.3)
        if val < best_val:
            best_val = val
            best_c = c.copy()
        scored.append((val, c))

    scored.sort(key=lambda t: t[0])
    top = [c for _, c in scored[:8]]
    if not top:
        top = [best_c]

    # Phase 2: refine top candidates with more iterations and smaller initial delta
    for x0 in top:
        if time.time() > _DEADLINE - 60:
            break
        c, val = _trust_region_sphere(Y_flat, w_flat, dphi, x0, maxiter=800, delta0=0.1)
        if val < best_val:
            best_val = val
            best_c = c.copy()

    # Phase 3: random restarts from perturbed best
    while time.time() < _DEADLINE - 20:
        x0 = best_c + 0.2 * rng.standard_normal(_DIM)
        x0 = _project_to_sphere(x0)
        c, val = _trust_region_sphere(Y_flat, w_flat, dphi, x0, maxiter=400, delta0=0.2)
        if val < best_val:
            best_val = val
            best_c = c.copy()

    out = best_c.astype(np.float64)
    out = out / np.linalg.norm(out)
    return out


# EVOLVE-BLOCK-END
