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
    """Return Y (nphi, nth, 2k+1), weights (nth,), and dphi."""
    k = _K
    mu, w = np.polynomial.legendre.leggauss(_NTH)
    theta = np.arccos(mu)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    Y = np.zeros((_NPHI, _NTH, 2 * k + 1), dtype=np.float64)
    for idx, m in enumerate(range(-k, k + 1)):
        am = abs(m)
        N = np.sqrt((2 * k + 1) / (4.0 * np.pi)
                    * float(factorial(k - am, exact=True))
                    / float(factorial(k + am, exact=True)))
        P = lpmv(am, k, mu)
        if m == 0:
            Y[:, :, idx] = N * P
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * N * P * np.cos(m * phi)[:, None]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * N * P * np.sin(am * phi)[:, None]
    return Y, w, dphi


def _objective(c, Y, w, dphi):
    """Compute R(c) = ||f||_4 / ||f||_2."""
    f = Y @ c  # shape: (nphi, nth)
    f2 = f * f
    f4 = f2 * f2
    # Integrate over theta (axis=1) and phi (axis=0)
    n2 = np.sum(f2 * w[None, :], axis=1)  # shape: (nphi,)
    n4 = np.sum(f4 * w[None, :], axis=1)  # shape: (nphi,)
    n2_val = np.sum(n2) * dphi
    n4_val = np.sum(n4) * dphi
    if n2_val < 1e-12:
        return 1e12
    return (n4_val ** 0.25) / np.sqrt(n2_val)


def _objective_and_grad(c, Y, w, dphi):
    """Compute R(c) and its gradient."""
    f = Y @ c  # shape: (nphi, nth)
    f2 = f * f
    f4 = f2 * f2
    n2 = np.sum(f2 * w[None, :], axis=1)  # shape: (nphi,)
    n4 = np.sum(f4 * w[None, :], axis=1)  # shape: (nphi,)
    n2_val = np.sum(n2) * dphi
    n4_val = np.sum(n4) * dphi
    if n2_val < 1e-12:
        return 1e12, np.zeros_like(c)
    
    R = (n4_val ** 0.25) / np.sqrt(n2_val)
    
    # Gradient of R = n4^(1/4) / n2^(1/2)
    # dR/dc = (1/4) n4^(-3/4) n2^(-1/2) dn4/dc - (1/2) n4^(1/4) n2^(-3/2) dn2/dc
    # dn2/dc = 2 * Y^T (f * w) * dphi
    # dn4/dc = 4 * Y^T (f^3 * w) * dphi
    
    # f * w[None, :] has shape (nphi, nth)
    # Y.T @ (f * w[None, :]) has shape (2k+1, nth)
    # We need to sum over nth axis to get (2k+1,)
    dn2_dc = 2.0 * np.sum(Y.transpose(2, 0, 1) * (f * w[None, :]), axis=(1, 2)) * dphi
    dn4_dc = 4.0 * np.sum(Y.transpose(2, 0, 1) * (f * f2 * w[None, :]), axis=(1, 2)) * dphi
    
    grad = (0.25 * (n4_val ** (-0.75)) * (n2_val ** (-0.5)) * dn4_dc
            - 0.5 * (n4_val ** 0.25) * (n2_val ** (-1.5)) * dn2_dc)
    
    return R, grad


def _slsqp_minimize(x0, Y, w, dphi, maxiter=1000):
    """Minimize R(c) subject to ||c||^2 = 1 using SLSQP with gradient."""
    
    def fun(c):
        val, _ = _objective_and_grad(c, Y, w, dphi)
        return val

    def jac(c):
        _, grad = _objective_and_grad(c, Y, w, dphi)
        return grad

    def cons(c):
        return np.sum(c * c) - 1.0

    def cons_jac(c):
        return 2.0 * c

    constraints = ({'type': 'eq', 'fun': cons, 'jac': cons_jac})
    
    try:
        res = minimize(fun, x0, jac=jac, method='SLSQP', constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.success:
            return res.x, res.fun
        else:
            # Try with the best point found
            best_c = x0.copy()
            best_val = _objective(x0, Y, w, dphi)
            # Check if res.x is better
            val_res = _objective(res.x, Y, w, dphi)
            if val_res < best_val:
                best_c = res.x
                best_val = val_res
            return best_c, best_val
    except Exception:
        return x0, _objective(x0, Y, w, dphi)


def _lbfgsb_minimize(x0, Y, w, dphi, maxiter=2000):
    """Minimize R(c) on the sphere using L-BFGS-B with reparameterization."""
    # Use a change of variables: c = v / ||v||
    # We optimize over v unconstrained, but project to sphere
    
    def fun_v(v):
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            return 1e12
        c = v / nrm
        val, _ = _objective_and_grad(c, Y, w, dphi)
        return val
    
    def jac_v(v):
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            return np.zeros_like(v)
        c = v / nrm
        _, grad_c = _objective_and_grad(c, Y, w, dphi)
        # Project gradient to tangent space
        proj = grad_c - c * np.dot(c, grad_c)
        return proj / nrm
    
    try:
        res = minimize(fun_v, x0, jac=jac_v, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'gtol': 1e-12})
        v_res = res.x
        nrm = np.linalg.norm(v_res)
        if nrm < 1e-12:
            return x0, _objective(x0, Y, w, dphi)
        c_res = v_res / nrm
        val_res = _objective(c_res, Y, w, dphi)
        
        val_x0 = _objective(x0, Y, w, dphi)
        if val_res < val_x0:
            return c_res, val_res
        else:
            return x0, val_x0
    except Exception:
        return x0, _objective(x0, Y, w, dphi)


def _make_inits(rng, k):
    """Generate diverse initializations."""
    inits = []
    dim = 2 * k + 1
    
    # 1. Single modes
    for m in range(-k, k + 1):
        v = np.zeros(dim)
        v[m + k] = 1.0
        inits.append(v)
        
    # 2. Pairs of conjugate modes (cos/sin)
    for m in range(1, k + 1):
        v = np.zeros(dim)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(v / np.linalg.norm(v))
        
    # 3. m=0 mixed with high m
    for m in [k, k-1, k-2]:
        for weight in [0.1, 0.3, 0.5, 0.7, 0.9]:
            v = np.zeros(dim)
            v[0 + k] = weight
            v[m + k] = np.sqrt(1 - weight**2)
            v[-m + k] = np.sqrt(1 - weight**2)
            inits.append(v / np.linalg.norm(v))
            
    # 4. Random
    for _ in range(40):
        v = rng.standard_normal(dim)
        inits.append(v / np.linalg.norm(v))
        
    # 5. Sparse random
    for _ in range(30):
        v = np.zeros(dim)
        idx = rng.choice(dim, size=rng.integers(2, 6), replace=False)
        v[idx] = rng.standard_normal(len(idx))
        inits.append(v / np.linalg.norm(v))
        
    # 6. Alternating signs
    v = np.ones(dim)
    v[::2] = -1.0
    inits.append(v / np.linalg.norm(v))
    
    # 7. More structured inits
    for m in range(1, k + 1):
        for phase in np.linspace(0, 2*np.pi, 4, endpoint=False):
            v = np.zeros(dim)
            v[m + k] = np.cos(phase)
            v[-m + k] = np.sin(phase)
            inits.append(v / np.linalg.norm(v))
    
    return inits


def rudin_problem10() -> np.ndarray:
    Y, w, dphi = _build_harmonics()
    k = _K
    rng = np.random.default_rng(42)
    inits = _make_inits(rng, k)

    best_c = inits[0].copy()
    best_val = _objective(best_c, Y, w, dphi)
    
    # Phase 1: Optimize from diverse starts using SLSQP with gradient
    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 120.0:
            break
        # Normalize start
        x0 = x0 / np.linalg.norm(x0)
        c, val = _slsqp_minimize(x0, Y, w, dphi, maxiter=500)
        if val < best_val:
            best_val = val
            best_c = c.copy()
            
    # Phase 2: Try L-BFGS-B with reparameterization from best
    if time.time() < _DEADLINE - 60.0:
        c, val = _lbfgsb_minimize(best_c, Y, w, dphi, maxiter=2000)
        if val < best_val:
            best_val = val
            best_c = c.copy()
            
    # Phase 3: Refine best with higher precision SLSQP
    if time.time() < _DEADLINE - 30.0:
        c, val = _slsqp_minimize(best_c, Y, w, dphi, maxiter=3000)
        if val < best_val:
            best_val = val
            best_c = c.copy()
            
    # Phase 4: Random restarts around best
    while time.time() < _DEADLINE - 15.0:
        x0 = best_c + 0.2 * rng.standard_normal(2 * k + 1)
        x0 = x0 / np.linalg.norm(x0)
        c, val = _slsqp_minimize(x0, Y, w, dphi, maxiter=300)
        if val < best_val:
            best_val = val
            best_c = c.copy()

    # Final polish
    if time.time() < _DEADLINE - 5.0:
        c, val = _slsqp_minimize(best_c, Y, w, dphi, maxiter=5000)
        if val < best_val:
            best_val = val
            best_c = c.copy()

    out = best_c.astype(np.float64)
    out = out / np.linalg.norm(out)
    return out


# EVOLVE-BLOCK-END
