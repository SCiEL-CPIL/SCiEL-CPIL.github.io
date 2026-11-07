"""Root-based parameterization with direct polynomial construction and multi-scale search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Exact Hermite values at 0
H0_0 = 1
H4_0 = 12
H8_0 = 1680
H12_0 = 2032128

# Coefficient matrix: rows correspond to z^0..z^5 in Q(z) = sum a_k z^k
# Columns correspond to (c0, c1, c2, c3)
# Q(z) = a2 + a4*z + a6*z^2 + a8*z^3 + a10*z^4 + a12*z^5
# where:
# a2  = -48*c1 - 13440*c2 - 1290240*c3
# a4  =  16*c1 + 11520*c2 + 2123328*c3
# a6  =      0*c2 - 3072*c2 - 1548288*c3  -> -3072*c2 - 1548288*c3
# a8  =      0*c2 +  256*c2 +  516096*c3
# a10 =      0*c2 +     0*c2 - 73728*c3
# a12 =      0*c2 +     0*c2 +  4096*c3

# Matrix M such that [a2,a4,a6,a8,a10,a12]^T = M @ [c0,c1,c2,c3]^T
M = np.array([
    [0,    -48,    -13440,  -1290240],
    [0,     16,    11520,   2123328],
    [0,      0,    -3072,  -1548288],
    [0,      0,      256,   516096],
    [0,      0,        0,   -73728],
    [0,      0,        0,    4096],
], dtype=float)

# Constraint: c3 = -(c0*1 + c1*12 + c2*1680) / 2032128
# So we can express everything in terms of c0, c1, c2 only.
# Let's precompute the mapping from (c0,c1,c2) to Q coefficients directly.

def compute_Q_coeffs(c0, c1, c2):
    """Compute the 6 coefficients [a2, a4, a6, a8, a10, a12] of Q(z)."""
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 1680.0) / 2032128.0
    
    a2  = -48.0*c1 - 13440.0*c2 - 1290240.0*c3
    a4  =  16.0*c1 + 11520.0*c2 + 2123328.0*c3
    a6  = -3072.0*c2 - 1548288.0*c3
    a8  =  256.0*c2 + 516096.0*c3
    a10 = -73728.0*c3
    a12 =  4096.0*c3
    
    return np.array([a2, a4, a6, a8, a10, a12])

def compute_value_from_Q(Q):
    """Given Q coefficients [a2,a4,a6,a8,a10,a12], compute the task value.
    Q(z) = a2 + a4*z + a6*z^2 + a8*z^3 + a10*z^4 + a12*z^5
    We need largest real root r of q(x)=Q(x^2) at which q changes sign.
    Equivalently, largest positive real root z of Q at which Q changes sign, then r=sqrt(z).
    value = r^2 / (2*pi) = z / (2*pi)
    
    Also check: if leading coeff a12 < 0, g -> -inf, so g is replaced by -g,
    which flips all Q coefficients.
    """
    Q = Q.copy()
    if Q[5] < 0:  # a12 < 0
        Q = -Q
    
    # Build polynomial for numpy.roots: highest degree first
    # Q(z) = a12*z^5 + a10*z^4 + a8*z^3 + a6*z^2 + a4*z + a2
    coeffs = np.array([Q[5], Q[4], Q[3], Q[2], Q[1], Q[0]])
    
    # Remove leading zeros
    idx = 0
    while idx < len(coeffs) - 1 and abs(coeffs[idx]) < 1e-15:
        idx += 1
    coeffs = coeffs[idx:]
    
    if len(coeffs) <= 1:
        return 1e10
    
    roots = np.roots(coeffs)
    
    # Filter: real, positive roots
    valid_z = []
    for rz in roots:
        if abs(rz.imag) < 1e-10:
            zr = rz.real
            if zr > 1e-12:
                valid_z.append(zr)
    
    if not valid_z:
        return 1e10
    
    # For each positive root, check sign change of Q
    best_z = 0.0
    for zr in valid_z:
        eps = 1e-8 * max(1.0, abs(zr))
        # Evaluate Q at zr - eps and zr + eps
        q_minus = np.polyval(coeffs, zr - eps)
        q_plus = np.polyval(coeffs, zr + eps)
        if q_minus * q_plus < 0:
            if zr > best_z:
                best_z = zr
    
    if best_z == 0.0:
        # Fallback: largest positive root
        best_z = max(valid_z)
    
    return best_z / (2.0 * np.pi)

def compute_value(c0, c1, c2):
    """Compute the task value from Hermite coefficients."""
    Q = compute_Q_coeffs(c0, c1, c2)
    return compute_value_from_Q(Q)

def objective(params):
    c0, c1, c2 = params
    try:
        v = compute_value(c0, c1, c2)
        return v if np.isfinite(v) else 1e10
    except:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin before 210s
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = compute_value(*seed)
    best_params = seed.copy()
    
    # Phase 1: Multi-start Nelder-Mead with diverse initial points
    np.random.seed(12345)
    
    starts = [seed.copy()]
    
    # Structured perturbations at multiple scales
    for scale in [0.01, 0.05, 0.1, 0.5, 1.0, 3.0]:
        for _ in range(4):
            pert = np.random.randn(3) * scale
            starts.append(seed + pert)
    
    # Some structured points in coefficient space
    for c0_val in [-1.5, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]:
        for c1_val in [-0.1, 0.0, 0.04, 0.1, 0.2]:
            starts.append(np.array([c0_val, c1_val, 0.0]))
    
    for start in starts:
        if time.time() > deadline - 40.0:
            break
        try:
            res = minimize(objective, start, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-11, 
                                   'fatol': 1e-13, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    # Phase 2: Local refinement around best found
    if time.time() < deadline - 20.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-14,
                                   'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    # Phase 3: Try BFGS if available and time permits
    if time.time() < deadline - 10.0:
        try:
            res = minimize(objective, best_params, method='L-BFGS-B',
                          options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    # Final polish
    if time.time() < deadline - 5.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-15,
                                   'fatol': 1e-17, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
