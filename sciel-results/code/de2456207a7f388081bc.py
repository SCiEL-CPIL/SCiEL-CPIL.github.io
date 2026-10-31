import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

T_START = time.time()
DEADLINE = 205.0

def _time_left():
    return DEADLINE - (time.time() - T_START)

def _hermite_at_0(n):
    """Compute H_n(0) using the recurrence H_n(0) = -2(n-1) H_{n-2}(0)."""
    if n == 0:
        return 1.0
    if n == 1:
        return 0.0
    # For even n: H_{2k}(0) = (-1)^k * (2k)! / k!
    # For odd n: H_n(0) = 0
    if n % 2 == 1:
        return 0.0
    k = n // 2
    # H_{2k}(0) = (-1)^k * (2k)! / k!
    import math
    return ((-1)**k) * math.factorial(2*k) / math.factorial(k)

def _compute_norm_sq(c):
    """Compute ||g||² = sum_i c_i² * 2^i * i! * sqrt(pi) where c = [c0, c4, c8, c12]."""
    import math
    n = np.array([0, 4, 8, 12])
    weights = 2.0**n * np.array([math.factorial(int(k)) for k in n]) * np.sqrt(np.pi)
    return np.sum(c**2 * weights)

def _compute_x4_norm_sq(c):
    """Compute ||x²g||² = ∫ x⁴ |g|² dx using analytic Hermite integrals."""
    import math
    n = np.array([0, 4, 8, 12])
    # ∫ x⁴ H_m H_n e^{-x²} dx = 2^n n! √π * [coefficients]
    # From our derivation:
    # x⁴ H_n = (1/16) H_{n+4} + (2n+3)/4 H_{n+2} + 3(2n²+2n+1)/4 H_n + n(n-1)(2n-1) H_{n-2} + n(n-1)(n-2)(n-3) H_{n-4}
    
    result = 0.0
    for i in range(4):
        for j in range(4):
            ni = n[i]
            nj = n[j]
            # We need ∫ x⁴ H_{ni} H_{nj} e^{-x²} dx
            # = 2^{nj} nj! √π * [ (1/16) δ_{ni,nj+4} + (2nj+3)/4 δ_{ni,nj+2} + 3(2nj²+2nj+1)/4 δ_{ni,nj} + nj(nj-1)(2nj-1) δ_{ni,nj-2} + nj(nj-1)(nj-2)(nj-3) δ_{ni,nj-4} ]
            
            coeff = 0.0
            if ni == nj + 4:
                coeff += 1.0/16.0
            if ni == nj + 2:
                coeff += (2*nj + 3.0)/4.0
            if ni == nj:
                coeff += 3.0*(2*nj**2 + 2*nj + 1.0)/4.0
            if ni == nj - 2:
                coeff += nj*(nj-1.0)*(2*nj-1.0)
            if ni == nj - 4:
                coeff += nj*(nj-1.0)*(nj-2.0)*(nj-3.0)
            
            if coeff != 0:
                integral = (2.0**nj) * math.factorial(int(nj)) * np.sqrt(np.pi) * coeff
                result += c[i] * c[j] * integral
    
    return result

def _compute_gpp_norm_sq(c):
    """Compute ||g''||² using analytic formulas."""
    import math
    # g''(x) = d(x) e^{-x²/2} where d(x) = sum_k d_k H_k(x)
    # d_k = 5(k+2)(k+1) c_{k+2} + (k - 1/2) c_k - (k+1) c_{k+1} - (1/2) c_{k-1} + (1/4) c_{k-2}
    # where c_j = 0 for j not in {0,4,8,12}
    
    # First, build the full coefficient array up to k=13
    c_full = {}
    for idx, deg in enumerate([0, 4, 8, 12]):
        c_full[deg] = c[idx]
    
    def get_c(j):
        if j < 0:
            return 0.0
        return c_full.get(j, 0.0)
    
    result = 0.0
    for k in range(0, 14):
        dk = 5.0*(k+2)*(k+1)*get_c(k+2) + (k - 0.5)*get_c(k) - (k+1.0)*get_c(k+1) - 0.5*get_c(k-1) + 0.25*get_c(k-2)
        if dk != 0.0:
            result += dk**2 * (2.0**k) * math.factorial(k) * np.sqrt(np.pi)
    
    return result

def _objective(c0_c4_c8):
    """Compute the uncertainty bound given (c0, c4, c8)."""
    c0, c4, c8 = c0_c4_c8
    
    # Compute c12 from constraint g(0) = 0
    h0_0 = _hermite_at_0(0)
    h4_0 = _hermite_at_0(4)
    h8_0 = _hermite_at_0(8)
    h12_0 = _hermite_at_0(12)
    
    c12 = -(c0*h0_0 + c4*h4_0 + c8*h8_0) / h12_0
    
    c = np.array([c0, c4, c8, c12])
    
    norm_sq = _compute_norm_sq(c)
    if norm_sq < 1e-15:
        return 1e10
    
    x4_sq = _compute_x4_norm_sq(c)
    gpp_sq = _compute_gpp_norm_sq(c)
    
    pos = np.sqrt(max(x4_sq / norm_sq, 0.0))
    freq = np.sqrt(max(gpp_sq / norm_sq, 0.0))
    
    return pos * freq

def _objective_grad(c0_c4_c8):
    """Compute objective and gradient using finite differences for robustness."""
    val = _objective(c0_c4_c8)
    if not np.isfinite(val):
        return val, np.zeros(3)
    
    eps = 1e-7
    grad = np.zeros(3)
    for i in range(3):
        c_plus = c0_c4_c8.copy()
        c_plus[i] += eps
        c_minus = c0_c4_c8.copy()
        c_minus[i] -= eps
        grad[i] = (_objective(c_plus) - _objective(c_minus)) / (2*eps)
    
    return val, grad

def construct():
    """Find optimal coefficients (c0, c4, c8) minimizing the uncertainty bound."""
    if _time_left() < 10.0:
        return np.array([-1.13, 0.04, 1.0/3240.0])
    
    # Try multiple starting points with L-BFGS-B
    starts = [
        np.array([-1.13, 0.04, 1.0/3240.0]),
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.5, 0.03, 0.0005]),
        np.array([-0.8, 0.06, 0.0002]),
        np.array([-1.2, 0.035, 0.00035]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-2.0, 0.1, 0.001]),
        np.array([-0.5, 0.1, 0.001]),
        np.array([-1.1, 0.045, 0.00031]),
        np.array([-1.15, 0.038, 0.00029]),
        np.array([-1.3, 0.04, 0.0003]),
        np.array([-0.9, 0.05, 0.0004]),
    ]
    
    best_val = np.inf
    best_x = starts[0].copy()
    
    for start in starts:
        if _time_left() < 5.0:
            break
        try:
            res = minimize(
                _objective,
                start,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Polish with tighter tolerance
    if _time_left() > 5.0 and np.isfinite(best_val):
        try:
            res = minimize(
                _objective,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x
