"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import hermite
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 200  # 200s safety margin within 210s limit

def _time_remaining():
    return _DEADLINE - time.time()

def _compute_value(c0, c1, c2):
    """Compute the value r^2/(2*pi) for given coefficients."""
    try:
        # Get Hermite polynomials (physicists')
        H0 = np.array([1.0])  # H_0(x) = 1
        H4 = np.array([16.0, 0.0, -48.0, 0.0, 24.0])  # H_4(x) = 16x^4 - 48x^2 + 12... wait
        # Let me compute properly using sympy-free approach
        # Physicists' Hermite: H_0=1, H_1=2x, H_2=4x^2-2, H_3=8x^3-12x, H_4=16x^4-48x^2+12
        # H_8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 560
        # H_12 = 4096x^12 - 114688x^10 + 1336320x^8 - 6126720x^6 + 11010048x^4 - 6935040x^2 + 1034880
        
        H0_coeffs = [1.0]  # constant
        H4_coeffs = [12.0, 0.0, -48.0, 0.0, 16.0]  # ascending: 12 + 0x - 48x^2 + 0x^3 + 16x^4
        H8_coeffs = [560.0, 0.0, -13440.0, 0.0, 11520.0, 0.0, -3072.0, 0.0, 256.0]
        H12_coeffs = [1034880.0, 0.0, -6935040.0, 0.0, 11010048.0, 0.0, -6126720.0, 0.0, 1336320.0, 0.0, -114688.0, 0.0, 4096.0]
        
        # g = c0*H0 + c1*H4 + c2*H8 + c3*H12
        # g(0) = c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
        H0_0 = 1.0
        H4_0 = 12.0
        H8_0 = 560.0
        H12_0 = 1034880.0
        
        c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
        
        # Build g coefficients (ascending order)
        # Pad to same length
        max_deg = 12
        g_coeffs = np.zeros(max_deg + 1)
        
        for i, c in enumerate(H0_coeffs):
            g_coeffs[i] += c0 * c
        for i, c in enumerate(H4_coeffs):
            g_coeffs[i] += c1 * c
        for i, c in enumerate(H8_coeffs):
            g_coeffs[i] += c2 * c
        for i, c in enumerate(H12_coeffs):
            g_coeffs[i] += c3 * c
        
        # g should be divisible by x^2 (g(0)=0 and g'(0)=0)
        # q(x) = g(x) / x^2, so q_coeffs = g_coeffs[2:]
        q_coeffs = g_coeffs[2:]
        
        # Find real roots of q
        roots = np.roots(q_coeffs)
        real_roots = roots[np.abs(roots.imag) < 1e-10].real
        
        # Find sign-changing roots: q changes sign at the root
        # For a root r, check if q(r-eps) and q(r+eps) have opposite signs
        sign_changing_roots = []
        for r in real_roots:
            if r <= 0:
                continue
            eps = 1e-8 * max(1.0, abs(r))
            q_left = np.polyval(q_coeffs, r - eps)
            q_right = np.polyval(q_coeffs, r + eps)
            if q_left * q_right < 0:
                sign_changing_roots.append(r)
        
        if not sign_changing_roots:
            # If no sign-changing roots, the value might be 0 or we need to handle differently
            # Check if g tends to negative limit at infinity - if so, flip
            leading_coeff = g_coeffs[-1]
            if leading_coeff < 0:
                # Flip g, which flips q
                q_coeffs = -q_coeffs
                # Re-check sign changing
                for r in real_roots:
                    if r <= 0:
                        continue
                    eps = 1e-8 * max(1.0, abs(r))
                    q_left = np.polyval(q_coeffs, r - eps)
                    q_right = np.polyval(q_coeffs, r + eps)
                    if q_left * q_right < 0:
                        sign_changing_roots.append(r)
            
            if not sign_changing_roots:
                # No sign-changing positive roots - value is 0 (optimal!)
                return 0.0
            r_max = max(sign_changing_roots)
        else:
            # Check leading coefficient for sign
            leading_coeff = g_coeffs[-1]
            if leading_coeff < 0:
                q_coeffs = -q_coeffs
                sign_changing_roots = []
                for r in real_roots:
                    if r <= 0:
                        continue
                    eps = 1e-8 * max(1.0, abs(r))
                    q_left = np.polyval(q_coeffs, r - eps)
                    q_right = np.polyval(q_coeffs, r + eps)
                    if q_left * q_right < 0:
                        sign_changing_roots.append(r)
                if not sign_changing_roots:
                    return 0.0
            r_max = max(sign_changing_roots)
        
        return r_max ** 2 / (2.0 * math.pi)
    except Exception:
        return float('inf')

def _objective(coeffs):
    """Objective function for optimization."""
    c0, c1, c2 = coeffs
    return _compute_value(c0, c1, c2)

def construct():
    best_value = _compute_value(-113/100, 1/25, 1/3240)
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    
    # Try multiple starting points
    start_points = [
        np.array([-1.13, 0.04, 0.00030864]),  # seed
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.2, 0.03, 0.0004]),
        np.array([-0.9, 0.06, 0.0002]),
        np.array([-1.13, 0.04, 0.0003]),
    ]
    
    for x0 in start_points:
        if _time_remaining() < 10:
            break
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Local refinement with tighter tolerances
    if _time_remaining() > 5:
        try:
            result = minimize(_objective, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Return the best coefficients found
    if _time_remaining() < 0:
        return np.array([-113/100, 1/25, 1/3240])
    
    return best_coeffs
# EVOLVE-BLOCK-END
