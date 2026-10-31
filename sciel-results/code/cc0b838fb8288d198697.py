"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    """Precompute I_n for n=0..13"""
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    # Use recurrence: I_n = (2n-1) * I_{n-1} / (4*n) for n>=1, I_0 = sqrt(pi)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) * I[n-1] / (4.0 * n)
    return I

_I = _precompute_I()


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    p = np.array([
        a0 + 3.0*a4 + 105.0*a8 + 945.0*a12,
        -6.0*a4 - 420.0*a8 - 7560.0*a12,
        a4 + 210.0*a8 + 13860.0*a12,
        -28.0*a8 - 6930.0*a12,
        a8 + 1155.0*a12,
        -66.0*a12,
        a12
    ])
    
    I = _I
    
    # m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j] for k=0,1,2
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        if pi == 0:
            continue
        for j in range(7):
            pp = pi * p[j]
            if pp == 0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    m[k] += pp * I[idx]
    
    # For Fourier transform: Q(xi) = sum q_k xi^{2k} where q_k = p_k * (-1)^k
    q = np.array([p[k] * ((-1.0)**k) for k in range(7)])
    
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for i in range(7):
        qi = q[i]
        if qi == 0:
            continue
        for j in range(7):
            qq = qi * q[j]
            if qq == 0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    n[k] += 2.0*np.pi * qq * I[idx]
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient analytically
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # Precompute sums for gradient: S_p[k+i] = sum_j p_j I[k+i+j]
    S_p = np.zeros(10)  # indices 0..9
    for i in range(7):
        for k in range(3):
            s = 0.0
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    s += p[j] * I[idx]
            S_p[k*7+i] = s  # flatten: index = k*7+i
    
    S_q = np.zeros(10)
    for i in range(7):
        for k in range(3):
            s = 0.0
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    s += q[j] * I[idx]
            S_q[k*7+i] = s
    
    grad = np.zeros(3)
    for c in range(3):
        dp = np.array([
            np.array([1.0, 3.0, 105.0])[c] + 945.0*da12[c],
            np.array([0.0, -6.0, -420.0])[c] - 7560.0*da12[c],
            np.array([0.0, 1.0, 210.0])[c] + 13860.0*da12[c],
            np.array([0.0, 0.0, -28.0])[c] - 6930.0*da12[c],
            np.array([0.0, 0.0, 1.0])[c] + 1155.0*da12[c],
            np.array([0.0, 0.0, 0.0])[c] - 66.0*da12[c],
            np.array([0.0, 0.0, 0.0])[c] + 1.0*da12[c]
        ])
        
        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0:
                continue
            for k in range(3):
                dm[k] += 2.0 * dpi * S_p[k*7+i]
        
        dq = dp * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        
        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0:
                continue
            for k in range(3):
                dn[k] += 2.0 * 2.0*np.pi * dqi * S_q[k*7+i]
        
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    # Multiple diverse starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0002]),
        np.array([-1.3, 0.05, 0.0004]),
        np.array([-1.7, 0.06, 0.0005]),
        np.array([-2.5, 0.12, 0.0015]),
        np.array([0.5, -0.02, 0.0001]),
        np.array([-0.3, 0.01, 0.0001]),
    ]
    
    rng = np.random.default_rng(42)
    # Add more diverse random starts to cover the landscape better for root finding
    for i in range(20):
        starts.append(rng.normal(0, 1.5, 3))
    
    best_val = np.inf
    best_params = None
    
    # Helper to evaluate and update best
    def try_update(params):
        nonlocal best_val, best_params
        if params is None:
            return
        try:
            val, _ = _bound_and_grad(params)
            if val < best_val and np.isfinite(val):
                best_val = val
                best_params = params.copy()
        except Exception:
            pass
    
    # 1. Multi-start Root Finding (Hybrid Powell)
    # We look for zeros of the gradient.
    # This is often more robust to "flat" regions or local minima traps than gradient descent
    # if the landscape has multiple basins.
    for start in starts:
        if time.time() > _DEADLINE - 40:
            break
        try:
            # Use root to find where gradient is zero
            res = root(_objective_grad, start, method='hybr', options={'xtol': 1e-14, 'maxfev': 2000})
            if res.success:
                try_update(res.x)
            else:
                # Even if root finding doesn't fully converge, the last iteration might be close
                # or we can try minimizing from there
                pass
        except Exception:
            pass
    
    # 2. If we found a candidate, polish it with L-BFGS-B to ensure it's a local minimum
    #    and to get high precision.
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    # 3. Final ultra-precision polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-20, 'gtol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
