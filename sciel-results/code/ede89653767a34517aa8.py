"""
Optimized uncertainty bound using scale-invariant 2D reparameterization 
and derivative-free Nelder-Mead search with analytic gradient polish.

Structural Change:
1. Exploits the scale invariance of the uncertainty product A(f)A(F) to reduce
   the search space from 3 coefficients (a0, a4, a8) to 2 ratios (x=a4/a0, y=a8/a0),
   fixing a0=1. This removes the ill-conditioning associated with the large dynamic
   range of the coefficients.
2. Uses Nelder-Mead (derivative-free) for the primary search, which is robust to
   the local landscape features that may trap gradient-based methods like L-BFGS-B
   in shallow local minima or stagnation points.
3. Polishes the best result with L-BFGS-B using the analytic gradient for high precision.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 205  # 5s safety margin before 210s kill

# Precompute I[n] = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
# We need n up to 12 (since max power in |P|^2 is x^{24}, so 2*12)
_I_CACHE = None

def _get_I():
    global _I_CACHE
    if _I_CACHE is not None:
        return _I_CACHE
    sp = np.sqrt(np.pi)
    I = np.zeros(15)
    # Use log-gamma or direct product for stability
    # (2n)! / (4^n n!) * sqrt(pi)
    # For small n, direct computation is fine
    fact = [1.0] * 30
    for i in range(1, 30):
        fact[i] = fact[i-1] * i
    for n in range(15):
        # (2n)! / (4^n * n!) * sqrt(pi)
        num = fact[2*n]
        den = (4.0**n) * fact[n]
        I[n] = (num / den) * sp
    _I_CACHE = I
    return I

_I = _get_I()

def _bound_and_grad_3d(params):
    """
    Compute uncertainty product and its analytic gradient for 3D params (a0, a4, a8).
    Used for the final polish.
    """
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
    
    m0 = 0.0
    m2 = 0.0
    m4 = 0.0
    for i in range(7):
        pi_val = p[i]
        for j in range(7):
            prod = pi_val * p[j]
            m0 += prod * _I[i+j]
            m2 += prod * _I[i+j+1]
            m4 += prod * _I[i+j+2]
    
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    n0 = 0.0
    n2 = 0.0
    n4 = 0.0
    for i in range(7):
        qi_val = q[i]
        for j in range(7):
            prod = qi_val * q[j]
            n0 += prod * _I[i+j]
            n2 += prod * _I[i+j+1]
            n4 += prod * _I[i+j+2]
    n0 *= 2.0 * np.pi
    n2 *= 2.0 * np.pi
    n4 *= 2.0 * np.pi
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    grad = np.zeros(3)
    for c in range(3):
        dp = np.zeros(7)
        dp[0] = np.array([1.0, 3.0, 105.0])[c] + 945.0*da12[c]
        dp[1] = np.array([0.0, -6.0, -420.0])[c] - 7560.0*da12[c]
        dp[2] = np.array([0.0, 1.0, 210.0])[c] + 13860.0*da12[c]
        dp[3] = np.array([0.0, 0.0, -28.0])[c] - 6930.0*da12[c]
        dp[4] = np.array([0.0, 0.0, 1.0])[c] + 1155.0*da12[c]
        dp[5] = np.array([0.0, 0.0, 0.0])[c] - 66.0*da12[c]
        dp[6] = np.array([0.0, 0.0, 0.0])[c] + 1.0*da12[c]
        
        dm0 = 0.0
        dm2 = 0.0
        dm4 = 0.0
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0:
                continue
            s0 = 0.0
            s2 = 0.0
            s4 = 0.0
            for j in range(7):
                s0 += p[j] * _I[i+j]
                s2 += p[j] * _I[i+j+1]
                s4 += p[j] * _I[i+j+2]
            dm0 += 2.0 * dpi * s0
            dm2 += 2.0 * dpi * s2
            dm4 += 2.0 * dpi * s4
        
        dq = dp * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        
        dn0 = 0.0
        dn2 = 0.0
        dn4 = 0.0
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0:
                continue
            s0 = 0.0
            s2 = 0.0
            s4 = 0.0
            for j in range(7):
                s0 += q[j] * _I[i+j]
                s2 += q[j] * _I[i+j+1]
                s4 += q[j] * _I[i+j+2]
            dn0 += 2.0 * dqi * s0
            dn2 += 2.0 * dqi * s2
            dn4 += 2.0 * dqi * s4
        dn0 *= 2.0 * np.pi
        dn2 *= 2.0 * np.pi
        dn4 *= 2.0 * np.pi
        
        dvm2 = (dm2*m0 - m2*dm0) / (m0*m0)
        dvn2 = (dn2*n0 - n2*dn0) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad

def _bound_2d(xy):
    """
    Compute uncertainty product for 2D params (x=a4/a0, y=a8/a0) with a0=1.
    This is the scale-invariant version.
    """
    x, y = xy[0], xy[1]
    a0 = 1.0
    a4 = x
    a8 = y
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    p0 = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p1 = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p2 = a4 + 210.0*a8 + 13860.0*a12
    p3 = -28.0*a8 - 6930.0*a12
    p4 = a8 + 1155.0*a12
    p5 = -66.0*a12
    p6 = a12
    
    p = np.array([p0, p1, p2, p3, p4, p5, p6])
    
    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        pi_val = p[i]
        for j in range(7):
            prod = pi_val * p[j]
            m0 += prod * _I[i+j]
            m2 += prod * _I[i+j+1]
    
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    n0 = 0.0
    n2 = 0.0
    for i in range(7):
        qi_val = q[i]
        for j in range(7):
            prod = qi_val * q[j]
            n0 += prod * _I[i+j]
            n2 += prod * _I[i+j+1]
    n0 *= 2.0 * np.pi
    n2 *= 2.0 * np.pi
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18
    
    return np.sqrt(vf * vF)

def _objective_3d(params):
    val, _ = _bound_and_grad_3d(params)
    return val

def _objective_grad_3d(params):
    val, grad = _bound_and_grad_3d(params)
    return val, grad

def construct():
    rng = np.random.default_rng(42)
    
    # The seed solution in (x, y) space:
    # a0 = -1.13, a4 = 0.04, a8 = 0.000308642
    # x = a4/a0 = 0.04 / -1.13 approx -0.0354
    # y = a8/a0 = 0.000308642 / -1.13 approx -0.000273
    seed_xy = np.array([0.04 / -1.13, 0.000308642 / -1.13])
    
    # Generate diverse starting points in (x, y) space
    starts = [seed_xy]
    
    # Local variations
    for i in range(50):
        starts.append(seed_xy + rng.normal(0, [0.01, 1e-4], 2))
    
    # Broader random starts
    for i in range(50):
        # x is likely small, y is very small
        # Use log-uniform for scale? Or just normal with wide range
        x = rng.uniform(-0.2, 0.2)
        y = rng.uniform(-0.002, 0.002)
        starts.append(np.array([x, y]))
    
    # Even broader
    for i in range(20):
        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-0.01, 0.01)
        starts.append(np.array([x, y]))
    
    best_val = np.inf
    best_xy = None
    
    # Phase 1: Nelder-Mead on 2D space
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_bound_2d, start, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_xy = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Polish with L-BFGS-B on 3D space
    # Map best_xy back to 3D: a0=1, a4=x, a8=y
    if best_xy is not None:
        best_3d = np.array([1.0, best_xy[0], best_xy[1]])
        
        # Try polishing from this point
        if time.time() < _DEADLINE - 10:
            try:
                res = minimize(_objective_grad_3d, best_3d, method='L-BFGS-B',
                              jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_3d = res.x.copy()
            except Exception:
                pass
        
        # Also try polishing from the original seed 3D point, in case 2D search missed something
        # The original seed 3D point
        seed_3d = np.array([-1.13, 0.04, 0.000308642])
        if time.time() < _DEADLINE - 5:
            try:
                res = minimize(_objective_grad_3d, seed_3d, method='L-BFGS-B',
                              jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_3d = res.x.copy()
            except Exception:
                pass
        
        # Return the best 3D parameters found
        if best_3d is not None:
            return np.array(best_3d)
    
    # Fallback to seed
    return np.array([-1.13, 0.04, 0.000308642])
# EVOLVE-BLOCK-END
