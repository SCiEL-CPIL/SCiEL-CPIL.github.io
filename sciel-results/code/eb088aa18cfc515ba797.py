"""Uncertainty bound minimization via global grid search + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200


def _precompute():
    """Precompute constant arrays for fast bound evaluation."""
    sp = np.sqrt(np.pi)
    # I[n] = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
    I = np.zeros(14)
    I[0] = sp
    for n in range(1, 14):
        log_val = 0.0
        for m in range(1, 2*n+1):
            log_val += np.log(m)
        log_val -= n * np.log(4.0)
        for m in range(1, n+1):
            log_val -= np.log(m)
        log_val += 0.5 * np.log(np.pi)
        I[n] = np.exp(log_val)
    
    # Base dp_k/da_c matrix (without a12 contribution)
    base_dp = np.array([
        [1.0, 3.0, 105.0],
        [0.0, -6.0, -420.0],
        [0.0, 1.0, 210.0],
        [0.0, 0.0, -28.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    
    dp_da12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
    da12_da_c = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    full_dp = np.zeros((7, 3))
    for c in range(3):
        full_dp[:, c] = base_dp[:, c] + dp_da12 * da12_da_c[c]
    
    return I, full_dp, da12_da_c


_I, _full_dp, _da12_da_c = _precompute()


def _bound(params):
    """Compute uncertainty product A(f)*A(F)."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    p0 = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p1 = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p2 = a4 + 210.0*a8 + 13860.0*a12
    p3 = -28.0*a8 - 6930.0*a12
    p4 = a8 + 1155.0*a12
    p5 = -66.0*a12
    p6 = a12
    
    p = np.array([p0, p1, p2, p3, p4, p5, p6])
    I = _I
    
    m = np.zeros(3)
    for k in range(3):
        for i in range(7):
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    m[k] += p[i] * p[j] * I[idx]
    
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0)**k)
    
    n = np.zeros(3)
    for k in range(3):
        for i in range(7):
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    n[k] += 2.0*np.pi * q[i] * q[j] * I[idx]
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    bound = np.sqrt(vf * vF)
    if not np.isfinite(bound):
        return 1e18
    
    return bound


def _bound_and_grad(params):
    """Compute uncertainty product and its analytic gradient."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    p0 = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p1 = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p2 = a4 + 210.0*a8 + 13860.0*a12
    p3 = -28.0*a8 - 6930.0*a12
    p4 = a8 + 1155.0*a12
    p5 = -66.0*a12
    p6 = a12
    
    p = np.array([p0, p1, p2, p3, p4, p5, p6])
    I = _I
    
    m = np.zeros(3)
    for k in range(3):
        for i in range(7):
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    m[k] += p[i] * p[j] * I[idx]
    
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0)**k)
    
    n = np.zeros(3)
    for k in range(3):
        for i in range(7):
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    n[k] += 2.0*np.pi * q[i] * q[j] * I[idx]
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    if not np.isfinite(bound):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    grad = np.zeros(3)
    for c in range(3):
        dp = _full_dp[:, c]
        
        dm = np.zeros(3)
        for k in range(3):
            s = 0.0
            for i in range(7):
                dpi = dp[i]
                if dpi == 0.0:
                    continue
                inner = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        inner += p[j] * I[idx]
                s += dpi * inner
            dm[k] = 2.0 * s
        
        dq = np.zeros(7)
        for k in range(7):
            dq[k] = dp[k] * ((-1.0)**k)
        
        dn = np.zeros(3)
        for k in range(3):
            s = 0.0
            for i in range(7):
                dqi = dq[i]
                if dqi == 0.0:
                    continue
                inner = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        inner += q[j] * I[idx]
                s += dqi * inner
            dn[k] = 4.0 * np.pi * s
        
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    if not np.all(np.isfinite(grad)):
        grad = np.array([0.0, 0.0, 0.0])
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    """
    Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce
    the upper bound on A(f)*A(Fourier(f)).
    
    Returns three coefficients (c0, c4, c8) of the Hermite combination.
    The degree-12 weight is fixed so g(0) = 0.
    """
    start_time = time.time()
    deadline = start_time + 180  # Safety margin
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: Global grid search over a2D parameterization
    # We parameterize as: a0 = u, a4 = v, a8 = 0, a12 = -(u+3v)/945
    # But we also need to explore a8, so we do a 3D grid but coarser
    # Actually, let's do a structured search: fix a8 at several values,
    # then grid over (a0, a4)
    
    a8_values = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    a0_range = np.linspace(-5.0, 5.0, 40)
    a4_range = np.linspace(-0.5, 0.5, 40)
    
    grid_best_val = np.inf
    grid_best_params = None
    
    for a8 in a8_values:
        if time.time() > deadline - 60:
            break
        for a0 in a0_range:
            for a4 in a4_range:
                params = np.array([a0, a4, a8])
                val = _bound(params)
                if val < grid_best_val:
                    grid_best_val = val
                    grid_best_params = params.copy()
    
    if grid_best_params is not None:
        best_val = grid_best_val
        best_params = grid_best_params
    
    # Phase 2: Local refinement from grid best using L-BFGS-B
    if best_params is not None and time.time() < deadline - 30:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Multi-start L-BFGS-B from diverse initial points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0003]),
        np.array([-1.2, 0.06, 0.0006]),
        np.array([2.0, -0.1, 0.001]),
        np.array([-4.0, 0.2, 0.005]),
        np.array([0.5, -0.02, 0.0001]),
        np.array([-2.5, 0.12, 0.0015]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(40):
        starts.append(rng.normal(0, 1.0, 3))
    
    for start in starts:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Polish with Nelder-Mead
    if best_params is not None and time.time() < deadline - 10:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final fine-tuning with very tight tolerances
    if best_params is not None and time.time() < deadline - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is not None:
        return best_params
    return np.array([-1.13, 0.04, 0.000308642])

# EVOLVE-BLOCK-END
