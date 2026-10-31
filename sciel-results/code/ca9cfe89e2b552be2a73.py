import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195.0

def _check_deadline():
    return time.time() < _DEADLINE

# Precompute moments J[n] = integral_{-inf}^{inf} x^{2n} e^{-x^2} dx
def _precompute_moments(max_n=13):
    J = np.zeros(max_n + 1)
    J[0] = np.sqrt(np.pi)
    for n in range(1, max_n + 1):
        J[n] = (2.0 * n - 1.0) / 2.0 * J[n - 1]
    return J

_J = _precompute_moments(13)

# Hermite polynomial coefficients: H_n(x) = sum_{k} h[n,k] x^{2k}
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 4480x^6 + 26880x^4 - 48384x^2 + 1680
# H_12(x) = 4096x^12 - 129024x^10 + 1548288x^8 - 7654656x^6 + 16636928x^4 - 15974400x^2 + 665280

# Coefficients for each Hermite polynomial: h[n, k] is coefficient of x^{2k}
# Index: n in {0, 1, 2, 3} corresponds to H_0, H_4, H_8, H_12
# k in 0..6 corresponds to x^0, x^2, x^4, x^6, x^8, x^10, x^12
_H_COEFFS = np.zeros((4, 7))
# H_0
_H_COEFFS[0, 0] = 1.0
# H_4: 16x^4 - 48x^2 + 12
_H_COEFFS[1, 0] = 12.0
_H_COEFFS[1, 1] = -48.0
_H_COEFFS[1, 2] = 16.0
# H_8: 256x^8 - 4480x^6 + 26880x^4 - 48384x^2 + 1680
_H_COEFFS[2, 0] = 1680.0
_H_COEFFS[2, 1] = -48384.0
_H_COEFFS[2, 2] = 26880.0
_H_COEFFS[2, 3] = -4480.0
_H_COEFFS[2, 4] = 256.0
# H_12: 4096x^12 - 129024x^10 + 1548288x^8 - 7654656x^6 + 16636928x^4 - 15974400x^2 + 665280
_H_COEFFS[3, 0] = 665280.0
_H_COEFFS[3, 1] = -15974400.0
_H_COEFFS[3, 2] = 16636928.0
_H_COEFFS[3, 3] = -7654656.0
_H_COEFFS[3, 4] = 1548288.0
_H_COEFFS[3, 5] = -129024.0
_H_COEFFS[3, 6] = 4096.0

# Constraint: P(0) = 0 => c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
# H0(0) = 1, H4(0) = 12, H8(0) = 1680, H12(0) = 665280
# c12 = -(c0 + 12*c4 + 1680*c8) / 665280

def _get_p_coeffs(c):
    """Get the polynomial coefficients p[k] for P(x) = sum p[k] x^{2k}"""
    c0, c4, c8 = c
    c12 = -(c0 + 12.0*c4 + 1680.0*c8) / 665280.0
    coeffs = np.array([c0, c4, c8, c12])
    # p[k] = sum_n coeffs[n] * H_COEFFS[n, k]
    p = _H_COEFFS.T @ coeffs
    return p

def _compute_bound_and_grad(c):
    """Compute the variance (upper bound on C4) and its gradient with respect to c0, c4, c8."""
    c0, c4, c8 = c
    c12 = -(c0 + 12.0*c4 + 1680.0*c8) / 665280.0
    coeffs = np.array([c0, c4, c8, c12])
    
    # Get polynomial coefficients p[k] for x^{2k}
    p = _H_COEFFS.T @ coeffs
    
    # Compute m0 = integral P^2 e^{-x^2} dx and m2 = integral x^2 P^2 e^{-x^2} dx
    # m0 = sum_{i,j} p[i] p[j] J[(i+j)//2]
    # m2 = sum_{i,j} p[i] p[j] J[(i+j+2)//2]
    
    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        if p[i] == 0: continue
        for j in range(7):
            if p[j] == 0: continue
            n0 = (i + j) // 2
            n2 = (i + j + 2) // 2
            if n0 <= 13:
                m0 += p[i] * p[j] * _J[n0]
            if n2 <= 13:
                m2 += p[i] * p[j] * _J[n2]
    
    if m0 <= 0:
        return 1e18, np.zeros(3)
    
    variance = m2 / m0
    
    # Gradient computation
    # variance = m2 / m0
    # d(variance)/d(c_a) = (dm2/d(c_a) * m0 - m2 * dm0/d(c_a)) / m0^2
    
    # p[k] = sum_n coeffs[n] * H_COEFFS[n, k]
    # coeffs = [c0, c4, c8, c12] where c12 = -(c0 + 12*c4 + 1680*c8)/665280
    
    # dp/d(c0) = [1, 0, 0, -1/665280] @ H_COEFFS.T
    # dp/d(c4) = [0, 1, 0, -12/665280] @ H_COEFFS.T
    # dp/d(c8) = [0, 0, 1, -1680/665280] @ H_COEFFS.T
    
    dc12_dc0 = -1.0 / 665280.0
    dc12_dc4 = -12.0 / 665280.0
    dc12_dc8 = -1680.0 / 665280.0
    
    # For each free coefficient c_a in {c0, c4, c8}:
    # d(coeffs)/dc_a = e_a + (dc12_dc_a) * e_3
    
    grad = np.zeros(3)
    dc_dc = np.array([
        [1.0, 0.0, 0.0, dc12_dc0],
        [0.0, 1.0, 0.0, dc12_dc4],
        [0.0, 0.0, 1.0, dc12_dc8]
    ])  # 3x4 matrix: rows are c0, c4, c8; cols are coeffs [c0, c4, c8, c12]
    
    for a in range(3):
        # dp_a[k] = d(p[k])/dc_a = sum_n dc_dc[a, n] * H_COEFFS[n, k]
        dp_a = _H_COEFFS.T @ dc_dc[a]
        
        # dm0/dc_a = sum_{i,j} (dp_a[i] * p[j] + p[i] * dp_a[j]) * J[(i+j)//2]
        #           = 2 * sum_{i,j} dp_a[i] * p[j] * J[(i+j)//2]  (by symmetry)
        dm0_dc = 0.0
        dm2_dc = 0.0
        for i in range(7):
            if dp_a[i] == 0: continue
            for j in range(7):
                if p[j] == 0: continue
                n0 = (i + j) // 2
                n2 = (i + j + 2) // 2
                if n0 <= 13:
                    dm0_dc += 2.0 * dp_a[i] * p[j] * _J[n0]
                if n2 <= 13:
                    dm2_dc += 2.0 * dp_a[i] * p[j] * _J[n2]
        
        grad[a] = (dm2_dc * m0 - m2 * dm0_dc) / (m0 * m0)
    
    return variance, grad

def _objective(c):
    val, _ = _compute_bound_and_grad(c)
    return val

def _objective_grad(c):
    val, grad = _compute_bound_and_grad(c)
    return val, grad

def construct():
    start_time = time.time()
    
    # Multiple initial guesses to avoid local minima
    initial_guesses = [
        np.array([-1.13, 0.04, 1.0/3240.0]),
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.5, 0.03, 0.0005]),
        np.array([-0.8, 0.06, 0.0002]),
        np.array([-2.0, 0.02, 0.0008]),
        np.array([-1.0, 0.0, 0.001]),
        np.array([-0.5, 0.1, 0.0001]),
        np.array([-1.2, 0.05, 0.0004]),
    ]
    
    best_c = None
    best_val = np.inf
    
    for c_init in initial_guesses:
        if time.time() > _DEADLINE - 5:
            break
        try:
            res = minimize(_objective_grad, c_init, method='L-BFGS-B',
                         options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # If no valid result, fall back to seed
    if best_c is None or not np.isfinite(best_val):
        c_init = np.array([-1.13, 0.04, 1.0/3240.0])
        best_c = c_init.copy()
        best_val = _objective(c_init)
    
    return best_c
