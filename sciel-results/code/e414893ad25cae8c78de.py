"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_hermite_coeffs():
    """Compute coefficients of physicists' Hermite polynomials H_0, H_4, H_8, H_12.
    Returns dict mapping degree to array of coefficients (ascending power)."""
    # Recurrence: H_{n+1} = 2x*H_n - 2n*H_{n-1}
    H = {}
    H[0] = np.array([1.0])
    H[1] = np.array([0.0, 2.0])
    
    for n in range(1, 12):
        len_n = len(H[n])
        len_nm1 = len(H[n-1])
        max_len = max(len_n + 1, len_nm1)
        
        two_x_Hn = np.zeros(max_len)
        two_x_Hn[1:len_n+1] = 2.0 * H[n]
        
        two_n_Hnm1 = np.zeros(max_len)
        two_n_Hnm1[:len_nm1] = 2.0 * n * H[n-1]
        
        H[n+1] = two_x_Hn - two_n_Hnm1
    
    return {k: H[k] for k in [0, 4, 8, 12]}


# Precompute Hermite polynomial coefficients
_HERMITE_COEFFS = _compute_hermite_coeffs()
H0_c = _HERMITE_COEFFS[0]   # [1]
H4_c = _HERMITE_COEFFS[4]   # [12, -48, 16]
H8_c = _HERMITE_COEFFS[8]   # [1680, -13440, 13440, -3584, 256]
H12_c = _HERMITE_COEFFS[12] # [665280, -14155780608, 128381414912, -414219003904, 575508845056, -332869504512, 68719476736]

# Values at x=0 (constant terms)
H0_0 = H0_c[0]    # 1
H4_0 = H4_c[0]    # 12
H8_0 = H8_c[0]    # 1680
H12_0 = H12_c[0]  # 665280


def _compute_value(c0, c1, c2):
    """Compute the value = r^2 / (2*pi) for given coefficients."""
    # c3 so that g(0) = 0
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Build g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # g(x) is a polynomial in x with only even powers (all H_n for even n have only even powers)
    # g(x) = sum_{k=0}^{6} a_k * x^{2k}, with a_0 = 0 (by construction g(0)=0)
    # q(x) = g(x)/x^2 = sum_{k=0}^{5} a_{k+1} * x^{2k}
    # Let y = x^2, then Q(y) = sum_{k=0}^{5} a_{k+1} * y^k
    
    # Coefficients of g(x) for x^{2k}:
    # H0: [1] -> contributes to x^0
    # H4: [12, -48, 16] -> 12 + (-48)x^2 + 16x^4
    # H8: [1680, -13440, 13440, -3584, 256] -> 1680 + (-13440)x^2 + 13440x^4 + (-3584)x^6 + 256x^8
    # H12: [665280, -14155780608, 128381414912, -414219003904, 575508845056, -332869504512, 68719476736]
    #       -> 665280 + (-14155780608)x^2 + 128381414912x^4 + (-414219003904)x^6 + 575508845056x^8 + (-332869504512)x^10 + 68719476736x^12
    
    # g(x) coefficients for x^{2k}, k=0..6:
    a = np.zeros(7)
    a[0] = c0 * H0_c[0] + c1 * H4_c[0] + c2 * H8_c[0] + c3 * H12_c[0]  # should be 0
    a[1] = c1 * H4_c[1] + c2 * H8_c[1] + c3 * H12_c[1]
    a[2] = c1 * H4_c[2] + c2 * H8_c[2] + c3 * H12_c[2]
    a[3] = c2 * H8_c[3] + c3 * H12_c[3]
    a[4] = c2 * H8_c[4] + c3 * H12_c[4]
    a[5] = c3 * H12_c[5]
    a[6] = c3 * H12_c[6]
    
    # Q(y) = a[1] + a[2]*y + a[3]*y^2 + a[4]*y^3 + a[5]*y^4 + a[6]*y^5
    # Coefficients in descending order for np.roots:
    Q_coeffs = np.array([a[6], a[5], a[4], a[3], a[2], a[1]])
    
    # Find roots of Q(y) = 0
    roots_y = np.roots(Q_coeffs)
    
    # Filter real, non-negative roots
    real_roots = []
    for r_y in roots_y:
        if abs(r_y.imag) < 1e-10 and r_y.real >= -1e-10:
            real_roots.append(max(0.0, r_y.real))
    
    if not real_roots:
        return float('inf')
    
    # For each candidate root y, check if Q changes sign there
    def Q(y):
        return a[6]*y**5 + a[5]*y**4 + a[4]*y**3 + a[3]*y**2 + a[2]*y + a[1]
    
    best_r2 = 0
    for y_root in real_roots:
        if y_root < 1e-14:
            continue
        # Check sign change: evaluate Q at y_root - eps and y_root + eps
        eps = 1e-8 * max(1.0, abs(y_root))
        q_minus = Q(y_root - eps)
        q_plus = Q(y_root + eps)
        if q_minus * q_plus < 0:
            # Sign change, this is a valid root
            r2 = y_root
            if r2 > best_r2:
                best_r2 = r2
    
    if best_r2 == 0:
        return float('inf')
    
    return best_r2 / (2 * np.pi)


def _compute_value_and_active_root(c0, c1, c2):
    """Compute value and return the active root y for smooth optimization."""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    a = np.zeros(7)
    a[0] = c0 * H0_0 + c1 * H4_0 + c2 * H8_0 + c3 * H12_0
    a[1] = c1 * H4_c[1] + c2 * H8_c[1] + c3 * H12_c[1]
    a[2] = c1 * H4_c[2] + c2 * H8_c[2] + c3 * H12_c[2]
    a[3] = c2 * H8_c[3] + c3 * H12_c[3]
    a[4] = c2 * H8_c[4] + c3 * H12_c[4]
    a[5] = c3 * H12_c[5]
    a[6] = c3 * H12_c[6]
    
    Q_coeffs = np.array([a[6], a[5], a[4], a[3], a[2], a[1]])
    roots_y = np.roots(Q_coeffs)
    
    real_roots = []
    for r_y in roots_y:
        if abs(r_y.imag) < 1e-10 and r_y.real >= -1e-10:
            real_roots.append(max(0.0, r_y.real))
    
    if not real_roots:
        return float('inf'), None
    
    def Q(y):
        return a[6]*y**5 + a[5]*y**4 + a[4]*y**3 + a[3]*y**2 + a[2]*y + a[1]
    
    best_r2 = 0
    active_y = None
    for y_root in real_roots:
        if y_root < 1e-14:
            continue
        eps = 1e-8 * max(1.0, abs(y_root))
        q_minus = Q(y_root - eps)
        q_plus = Q(y_root + eps)
        if q_minus * q_plus < 0:
            if y_root > best_r2:
                best_r2 = y_root
                active_y = y_root
    
    if active_y is None:
        return float('inf'), None
    
    return best_r2 / (2 * np.pi), active_y


def construct():
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 210s kill
    
    # Seed coefficients
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_value = _compute_value(*seed)
    best_coeffs = seed.copy()
    best_value = seed_value
    
    random_state = np.random.RandomState(42)
    
    # Generate multiple starting points
    starts = [seed.copy()]
    for _ in range(15):
        perturb = random_state.randn(3) * 0.05
        starts.append(seed + perturb)
    for _ in range(10):
        starts.append(random_state.randn(3) * 0.3)
    
    # Phase 1: Nelder-Mead from multiple starts
    for i, start in enumerate(starts):
        if time.time() > deadline - 30.0:
            break
        
        def objective(x):
            return _compute_value(x[0], x[1], x[2])
        
        try:
            result = minimize(objective, start, method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-16,
                                     'adaptive': True})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B with numerical gradient from best point
    if time.time() < deadline - 20.0:
        def objective_l(x):
            return _compute_value(x[0], x[1], x[2])
        
        try:
            result = minimize(objective_l, best_coeffs, method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14,
                                     'maxls': 50})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish with Nelder-Mead (tighter)
    if time.time() < deadline - 10.0:
        def objective_polish(x):
            return _compute_value(x[0], x[1], x[2])
        
        try:
            result = minimize(objective_polish, best_coeffs, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-18, 'fatol': 1e-20,
                                     'adaptive': True})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Phase 4: One more L-BFGS-B polish if time permits
    if time.time() < deadline - 5.0:
        def objective_final(x):
            return _compute_value(x[0], x[1], x[2])
        
        try:
            result = minimize(objective_final, best_coeffs, method='L-BFGS-B',
                            options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-16,
                                     'maxls': 100})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
