"""
Combined approach: Analytic evaluation of the uncertainty bound using
Hermite polynomial identities, with global-to-local optimization on the 2-sphere.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import gamma
from math import factorial, sqrt, pi

_START = time.time()
_DEADLINE = _START + 175.0  # Safety margin under 210s

def _time_left():
    return _DEADLINE - time.time()

# Hermite polynomials (physicists') even ones, coefficients for x^0..x^12
# H0 = 1
# H4 = 16x^4 - 48x^2 + 12
# H8 = 256x^8 - 1280x^6 + 2688x^4 - 1344x^2 + 112
# H12 = 4096x^12 - 39936x^10 + 153600x^8 - 282240x^6 + 253440x^4 - 96768x^2 + 9216

def _b_coeffs(c0, c4, c8):
    """
    Compute the polynomial coefficients b[m] = coeff of x^{2m} in P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    where c12 is chosen so that P(0) = 0.
    """
    # P(0) = c0*1 + c4*12 + c8*112 + c12*9216 = 0
    c12 = -(c0 + 12.0*c4 + 112.0*c8) / 9216.0
    b = np.zeros(7)  # b[0]=b0 (x^0), b[1]=b2 (x^2), ..., b[6]=b12 (x^12)
    # x^0: c0*1 + c4*12 + c8*112 + c12*9216 = 0 (by construction)
    b[0] = 0.0
    # x^2: c4*(-48) + c8*(-1344) + c12*(-96768)
    b[1] = c4*(-48.0) + c8*(-1344.0) + c12*(-96768.0)
    # x^4: c4*16 + c8*2688 + c12*253440
    b[2] = c4*16.0 + c8*2688.0 + c12*253440.0
    # x^6: c8*(-1280) + c12*(-282240)
    b[3] = c8*(-1280.0) + c12*(-282240.0)
    # x^8: c8*256 + c12*153600
    b[4] = c8*256.0 + c12*153600.0
    # x^10: c12*(-39936)
    b[5] = c12*(-39936.0)
    # x^12: c12*4096
    b[6] = c12*4096.0
    return b

def _poly4_integral(coeffs_u):
    """
    Compute ∫ (Σ coeffs_u[m] u^m)^4 e^{-2x^2} dx where u = x^2.
    = Σ d[n] Γ(n+1/2)/2^{n+1/2} where d = coeffs_u * coeffs_u * coeffs_u * coeffs_u (4-fold convolution)
    """
    d = np.convolve(np.convolve(coeffs_u, coeffs_u), np.convolve(coeffs_u, coeffs_u))
    n = np.arange(len(d))
    vals = gamma(n + 0.5) / (2.0 ** (n + 0.5))
    return float(np.dot(d, vals))

# Precompute P_k polynomials (in ξ), P_k[i] = coeff of ξ^i
# P_0(ξ) = 1, P_{k+1}(ξ) = P_k'(ξ) - ξ P_k(ξ)
def _build_P():
    P = [[1.0]]  # P_0
    for k in range(12):
        Pk = P[k]
        # derivative
        dpk = [i*Pk[i] for i in range(1, len(Pk))]
        # x*Pk
        xPk = [0.0] + Pk
        L = max(len(dpk), len(xPk))
        dpk = dpk + [0.0]*(L-len(dpk))
        xPk = xPk + [0.0]*(L-len(xPk))
        Pk1 = [dpk[i] - xPk[i] for i in range(L)]
        P.append(Pk1)
    return P

_P = _build_P()

def _R_coeffs_u(b):
    """
    Compute R(ξ) = Σ_{j=0}^{6} b[j] * (-1)^j * P[2j](ξ)
    and return the even-coefficient form (in u = ξ^2).
    """
    R = np.zeros(13)  # up to ξ^12
    for j in range(7):
        sign = (-1.0)**j
        Pk = np.array(_P[2*j])
        R[:len(Pk)] += b[j]*sign*Pk
    # R is even; extract even coefficients -> u form
    R_u = R[0::2]  # R[0], R[2], ..., R[12]
    return R_u

def _raw_value(c0, c4, c8):
    """
    Compute the raw uncertainty bound A(f)*A(Fourier(f)) analytically.
    """
    b = _b_coeffs(c0, c4, c8)
    A_g = _poly4_integral(b)
    R_u = _R_coeffs_u(b)
    A_gf = _poly4_integral(R_u)
    return A_g * A_gf

_SEED = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
_RAW_SEED = _raw_value(*_SEED)

def _value(c0, c4, c8):
    return _raw_value(c0, c4, c8) / _RAW_SEED

def _objective(x):
    return _value(x[0], x[1], x[2])

def construct():
    global _START, _DEADLINE
    best_x = _SEED.copy()
    best_v = _value(*_SEED)  # = 1.0
    
    bounds = [(-10.0, 10.0), (-10.0, 10.0), (-5.0, 5.0)]
    
    try:
        # Phase 1: Differential evolution for global search
        if _time_left() > 40:
            res = differential_evolution(
                _objective, bounds, seed=42, maxiter=300, popsize=30,
                tol=1e-10, polish=False, init='sobol',
                update_interval=1,
                callback=lambda xk, convergence: convergence and (time.time() > _DEADLINE - 120)
            )
            if res.x is not None and res.fun < best_v:
                best_v = res.fun
                best_x = res.x.copy()
        
        # Phase 2: Local refinement with Nelder-Mead
        if _time_left() > 15:
            for x0 in [best_x, _SEED]:
                try:
                    r2 = minimize(_objective, x0, method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                    if r2.fun < best_v:
                        best_v = r2.fun
                        best_x = r2.x.copy()
                except Exception:
                    pass
        
        # Phase 3: Multi-start DE if time permits
        if _time_left() > 60:
            for s in [1, 2, 3, 4, 5]:
                if _time_left() < 40:
                    break
                try:
                    res = differential_evolution(_objective, bounds, seed=s, maxiter=200, popsize=25, tol=1e-10, polish=False)
                    if res.fun < best_v:
                        best_v = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass
        
        # Phase 4: Final high-precision polish
        if _time_left() > 5:
            try:
                r2 = minimize(_objective, best_x, method='L-BFGS-B', options={'maxiter': 10000, 'gtol': 1e-14})
                if r2.fun < best_v:
                    best_v = r2.fun
                    best_x = r2.x.copy()
            except Exception:
                pass
            
            # One more Nelder-Mead pass with very tight tolerances
            if _time_left() > 3:
                try:
                    r3 = minimize(_objective, best_x, method='Nelder-Mead',
                                  options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
                    if r3.fun < best_v:
                        best_v = r3.fun
                        best_x = r3.x.copy()
                except Exception:
                    pass
                    
    except Exception:
        pass
    
    return np.array([best_x[0], best_x[1], best_x[2]])
