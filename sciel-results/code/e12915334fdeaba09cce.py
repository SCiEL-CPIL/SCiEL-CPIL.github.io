"""Uncertainty bound minimization via matrix formulation and analytical gradients."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt, pi
from scipy.optimize import minimize

_DEADLINE = 205.0

# Precompute Hermite polynomial coefficients (low to high power), padded to length 13 (degree 12)
# H_n(x) = (-1)^n e^{x^2/2} d^n/dx^n e^{-x^2/2} ? No, physicists' Hermite.
# H_0 = 1
# H_4 = 16x^4 - 48x^2 + 12
# H_8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 2520
# H_12 = 4096x^12 - 92160x^10 + 645120x^8 - 1445760x^6 + 1290240x^4 - 399168x^2 + 27720

def _make_hermite_vec(coeffs):
    """Pad coefficients to length 13."""
    v = np.zeros(13)
    for i, c in enumerate(coeffs):
        v[i] = c
    return v

_H0 = _make_hermite_vec([1.0])
_H4 = _make_hermite_vec([12.0, 0.0, -48.0, 0.0, 16.0])
_H8 = _make_hermite_vec([2520.0, 0.0, -13440.0, 0.0, 11520.0, 0.0, -3072.0, 0.0, 256.0])
_H12 = _make_hermite_vec([27720.0, 0.0, -399168.0, 0.0, 1290240.0, 0.0, -1445760.0, 0.0, 645120.0, 0.0, -92160.0, 0.0, 4096.0])

# Values at 0
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 2520.0
_H12_0 = 27720.0

# Basis vectors satisfying g(0)=0
# g(x) = a0*u0(x) + a4*u4(x) + a8*u8(x)
# u0 = H0 - (1/27720)H12
# u4 = H4 - (12/27720)H12
# u8 = H8 - (2520/27720)H12
_u0 = _H0 - (_H0_0 / _H12_0) * _H12
_u4 = _H4 - (_H4_0 / _H12_0) * _H12
_u8 = _H8 - (_H8_0 / _H12_0) * _H12

_basis = np.array([_u0, _u4, _u8]) # Shape (3, 13)

def _int_x2m(m):
    """Integral of x^(2m) e^(-x^2)"""
    if m == 0:
        return sqrt(pi)
    val = sqrt(pi)
    for j in range(1, m + 1):
        val *= (2 * j - 1) / 2.0
    return val

def _poly_product(p1, p2):
    """Convolution of two coefficient vectors (low to high)."""
    return np.convolve(p1, p2)

def _int_poly_sq(p):
    """Integral of p(x)^2 e^(-x^2) dx"""
    p2 = _poly_product(p, p)
    res = 0.0
    for k in range(0, len(p2), 2):
        m = k // 2
        res += p2[k] * _int_x2m(m)
    return res

def _int_prod(p1, p2):
    """Integral of p1(x) p2(x) e^(-x^2) dx"""
    prod = _poly_product(p1, p2)
    res = 0.0
    for k in range(0, len(prod), 2):
        m = k // 2
        res += prod[k] * _int_x2m(m)
    return res

def _int_x2_prod(p1, p2):
    """Integral of x^2 p1(x) p2(x) e^(-x^2) dx"""
    prod = _poly_product(p1, p2)
    res = 0.0
    for k in range(0, len(prod), 2):
        m = (k + 2) // 2
        res += prod[k] * _int_x2m(m)
    return res

def _diff_minus_x(p):
    """Compute q(x) = p'(x) - x*p(x)"""
    n = len(p)
    q = np.zeros(n + 1)
    # p'(x)
    for k in range(1, n):
        q[k - 1] += p[k] * k
    # -x p(x)
    for k in range(n):
        q[k + 1] += -p[k]
    return q

# Precompute matrices
_A0 = np.zeros((3, 3))
_A2 = np.zeros((3, 3))
_AG = np.zeros((3, 3))

_q_basis = np.array([_diff_minus_x(b) for b in _basis])

for i in range(3):
    for j in range(3):
        _A0[i, j] = _int_prod(_basis[i], _basis[j])
        _A2[i, j] = _int_x2_prod(_basis[i], _basis[j])
        _AG[i, j] = _int_prod(_q_basis[i], _q_basis[j])

# Symmetrize
_A0 = (_A0 + _A0.T) / 2.0
_A2 = (_A2 + _A2.T) / 2.0
_AG = (_AG + _AG.T) / 2.0

def _objective_and_grad(v):
    """
    v: [a0, a4, a8]
    Returns (J, grad_J)
    J = (v^T A2 v * v^T AG v) / (v^T A0 v)^2
    """
    M0 = v @ _A0 @ v
    M2 = v @ _A2 @ v
    MG = v @ _AG @ v
    
    if M0 < 1e-15:
        return np.inf, np.zeros(3)
    
    J = (M2 * MG) / (M0 * M0)
    
    # Gradient
    # dJ = 2/M0^2 * [ MG * A2 v + M2 * AG v - 2 * (M2*MG/M0) * A0 v ]
    grad = (2.0 / (M0 * M0)) * (
        MG * (_A2 @ v) + 
        M2 * (_AG @ v) - 
        2.0 * (M2 * MG / M0) * (_A0 @ v)
    )
    
    return J, grad

def construct():
    start_time = time.time()
    
    best_v = np.array([0.0, 0.0, 0.0])
    best_val = np.inf
    
    # Multi-start optimization
    # The landscape might be non-convex, so we try several starts.
    # Based on physics intuition, a0 should be negative to cancel the constant term of H12?
    # H12(0) is large positive. H0(0)=1.
    # If a0 is negative, g(0) can be 0 with small a4, a8.
    
    starts = [
        np.array([-1.0, 0.0, 0.0]),
        np.array([-2.0, 0.0, 0.0]),
        np.array([-0.5, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.0]),
        np.array([-1.0, -0.1, 0.0]),
        np.array([-1.5, 0.0, 0.0]),
        np.array([-0.8, 0.0, 0.0]),
        np.array([-1.2, 0.0, 0.0]),
    ]
    
    for v0 in starts:
        if time.time() - start_time > _DEADLINE - 10:
            break
            
        try:
            res = minimize(
                lambda x: _objective_and_grad(x)[0],
                v0,
                jac=lambda x: _objective_and_grad(x)[1],
                method='BFGS',
                options={'gtol': 1e-10, 'maxiter': 200}
            )
            
            val, grad = _objective_and_grad(res.x)
            if val < best_val and np.isfinite(val):
                best_val = val
                best_v = res.x.copy()
        except Exception:
            continue

    return best_v
# EVOLVE-BLOCK-END
