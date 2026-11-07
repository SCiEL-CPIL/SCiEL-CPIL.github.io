import numpy as np
from scipy.optimize import minimize
import time

# Hermite polynomial coefficients (up to x^12)
# H0: 1
# H4: 12 - 48x^2 + 16x^4
# H8: 1680 - 13440x^2 + 11520x^4 - 3072x^6 + 256x^8
# H12: 10321920 - 5529600x^2 + 8709120x^4 - 4915200x^6 + 1032192x^8 - 98304x^10 + 4096x^12

H0_coeffs = np.array([1.0])
H4_coeffs = np.array([12.0, 0.0, -48.0, 0.0, 16.0])
H8_coeffs = np.array([1680.0, 0.0, -13440.0, 0.0, 11520.0, 0.0, -3072.0, 0.0, 256.0])
H12_coeffs = np.array([10321920.0, 0.0, -5529600.0, 0.0, 8709120.0, 0.0, -4915200.0, 0.0, 1032192.0, 0.0, -98304.0, 0.0, 4096.0])

def get_g_coeffs(c0, c1, c2, c3):
    # Pad to length 13
    g = np.zeros(13)
    g[0] += c0 * H0_coeffs[0]
    for i in range(len(H4_coeffs)):
        g[i] += c1 * H4_coeffs[i]
    for i in range(len(H8_coeffs)):
        g[i] += c2 * H8_coeffs[i]
    for i in range(len(H12_coeffs)):
        g[i] += c3 * H12_coeffs[i]
    return g

def get_q_coeffs(c0, c1, c2):
    # c3 is determined by g(0)=0 => c0*1 + c1*12 + c2*1680 + c3*10321920 = 0
    # c3 = -(c0 + 12c1 + 1680c2) / 10321920
    c3 = -(c0 + 12*c1 + 1680*c2) / 10321920.0
    g = get_g_coeffs(c0, c1, c2, c3)
    # q(x) = g(x)/x^2. g(0) should be 0.
    # q coeffs are g[2:]
    q = g[2:]
    # Remove trailing zeros to reduce degree if possible, though np.roots handles it
    # Find last non-zero
    while len(q) > 0 and abs(q[-1]) < 1e-12:
        q = q[:-1]
    if len(q) == 0:
        return np.array([1.0]) # Constant 1, no roots
    return q

def compute_value(c):
    c0, c1, c2 = c
    q_coeffs = get_q_coeffs(c0, c1, c2)
    
    # Find roots
    roots = np.roots(q_coeffs)
    
    # Filter real roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-9:
            real_roots.append(r.real)
            
    if not real_roots:
        return 1e10 # No real roots, bad
        
    # Check sign change for each real root
    # We want the LARGEST real root where sign changes.
    # Sign change: q(r-eps) * q(r+eps) < 0
    
    sign_changing_roots = []
    eps = 1e-6
    for r in real_roots:
        # Evaluate q at r-eps and r+eps
        # Use Horner's method or np.polyval
        val_minus = np.polyval(q_coeffs, r - eps)
        val_plus = np.polyval(q_coeffs, r + eps)
        
        if val_minus * val_plus < 0:
            sign_changing_roots.append(r)
            
    if not sign_changing_roots:
        return 1e10 # No sign changing roots
        
    r_max = max(sign_changing_roots)
    
    # Value is r^2 / (2*pi)
    return (r_max * r_max) / (2.0 * np.pi)

def construct():
    # Initial guess
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Time limit guard
    start_time = time.time()
    timeout = 180 # 180s limit, keep margin
    
    best_c = x0.copy()
    best_val = compute_value(best_c)
    
    # Try Nelder-Mead
    try:
        result = minimize(compute_value, x0, method='Nelder-Mead', 
                          options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < best_val:
            best_c = result.x
            best_val = result.fun
    except Exception:
        pass
        
    # Check time
    if time.time() - start_time > timeout * 0.8:
        return best_c
        
    # Try a few random restarts if time permits
    # The landscape might be multimodal
    np.random.seed(42)
    for _ in range(20):
        if time.time() - start_time > timeout * 0.9:
            break
        # Perturb initial guess
        x_rand = x0 + np.random.normal(0, 0.1, size=3)
        try:
            result = minimize(compute_value, x_rand, method='Nelder-Mead', 
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_val:
                best_c = result.x
                best_val = result.fun
        except Exception:
            pass

    return best_c
