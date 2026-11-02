"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

# Precompute constants
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 665280.0

def _hermite_polys(x):
    """Compute H0, H4, H8, H12 and their derivatives at x (array)."""
    x2 = x**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x6**2
    
    H0 = np.ones_like(x)
    H4 = 16*x4 - 48*x2 + 12
    H8 = 256*x8 - 4480*x6 + 26880*x4 - 6720*x2 + 1680
    H12 = 4096*x12 - 129024*x10 + 1761280*x8 - 11942400*x6 + 37158400*x4 - 54432000*x2 + 47900160
    
    # Derivatives: H_n'(x) = 2n H_{n-1}(x)
    # H3 = 8x^3 - 12x
    H3 = 8*x*x2 - 12*x
    # H7 = 128x^7 - 2240x^5 + 10080x^3 - 10080x
    H7 = 128*x*x6 - 2240*x*x4 + 10080*x*x2 - 10080*x
    # H11 = 2048x^11 - 57344x^9 + 645120x^7 - 3386880x^5 + 6978240x^3 - 4084032x
    # Note: H11(x) = 2*11*H10(x)/2? No.
    # Let's compute H11 explicitly or recursively.
    # H11 = 2048x^11 - 57344x^9 + 645120x^7 - 3386880x^5 + 6978240x^3 - 4084032x
    H11 = 2048*x*x10 - 57344*x*x8 + 645120*x*x6 - 3386880*x*x4 + 6978240*x*x2 - 4084032*x
    
    return H0, H4, H8, H12, H3, H7, H11

def _compute_bound_and_grad(c, N=400):
    """Compute bound and gradient."""
    c0, c4, c8 = c
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    x_gh, w_gh = np.polynomial.hermite.hermgauss(N)
    
    H0, H4, H8, H12, H3, H7, H11 = _hermite_polys(x_gh)
    
    # P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    P = c0*H0 + c4*H4 + c8*H8 + c12*H12
    P_prime = 2*(c4*H3 + 8*c8*H7 + 12*c12*H11)
    
    exp_term = np.exp(-x_gh**2 / 2.0)
    g_vals = P * exp_term
    f_vals = x_gh * g_vals
    
    # Find index of max |f|
    idx = np.argmax(np.abs(f_vals))
    x_star = x_gh[idx]
    f_star = f_vals[idx]
    
    # Refine x_star using Newton's method on f'(x) = 0
    # f(x) = x exp(-x^2/2) P(x)
    # f'(x) = exp(-x^2/2) [ P(x) + x P'(x) - x^2 P(x) ]
    # We need P and P' at x_star
    # Re-evaluate at x_star for precision
    H0_s, H4_s, H8_s, H12_s, H3_s, H7_s, H11_s = _hermite_polys(np.array([x_star]))
    H0_s, H4_s, H8_s, H12_s, H3_s, H7_s, H11_s = H0_s[0], H4_s[0], H8_s[0], H12_s[0], H3_s[0], H7_s[0], H11_s[0]
    
    P_s = c0*H0_s + c4*H4_s + c8*H8_s + c12*H12_s
    P_p_s = 2*(c4*H3_s + 8*c8*H7_s + 12*c12*H11_s)
    
    # Newton iterations
    for _ in range(10):
        # f'(x) = exp(-x^2/2) (P + x P' - x^2 P)
        # f''(x) is complex, use numerical derivative of f' or just one step?
        # Let's use numerical derivative of f_prime for Newton step
        def fp(x_val):
            H0_v, H4_v, H8_v, H12_v, H3_v, H7_v, H11_v = _hermite_polys(np.array([x_val]))
            H0_v, H4_v, H8_v, H12_v, H3_v, H7_v, H11_v = H0_v[0], H4_v[0], H8_v[0], H12_v[0], H3_v[0], H7_v[0], H11_v[0]
            P_v = c0*H0_v + c4*H4_v + c8*H8_v + c12*H12_v
            P_p_v = 2*(c4*H3_v + 8*c8*H7_v + 12*c12*H11_v)
            return math.exp(-x_val**2 / 2.0) * (P_v + x_val * P_p_v - x_val**2 * P_v)
        
        def fpp(x_val):
            h = 1e-6
            return (fp(x_val+h) - fp(x_val-h)) / (2*h)
        
        val = fp(x_star)
        if abs(val) < 1e-12: break
        der = fpp(x_star)
        if abs(der) < 1e-12: break
        x_star_new = x_star - val/der
        if abs(x_star_new - x_star) < 1e-10:
            x_star = x_star_new
            break
        x_star = x_star_new
        
    # Recalculate P and P' at refined x_star
    H0_s, H4_s, H8_s, H12_s, H3_s, H7_s, H11_s = _hermite_polys(np.array([x_star]))
    H0_s, H4_s, H8_s, H12_s, H3_s, H7_s, H11_s = H0_s[0], H4_s[0], H8_s[0], H12_s[0], H3_s[0], H7_s[0], H11_s[0]
    P_s = c0*H0_s + c4*H4_s + c8*H8_s + c12*H12_s
    P_p_s = 2*(c4*H3_s + 8*c8*H7_s + 12*c12*H11_s)
    
    exp_s = math.exp(-x_star**2 / 2.0)
    f_star = x_star * P_s * exp_s
    M = abs(f_star)
    bound = math.sqrt(2 * math.pi) * M * M
    
    # Gradient
    # dP/dc0 = H0_s - (1/H12_0) H12_s
    # dP/dc4 = H4_s - (12/H12_0) H12_s
    # dP/dc8 = H8_s - (1680/H12_0) H12_s
    dP_dc0 = H0_s - (1.0/H12_0) * H12_s
    dP_dc4 = H4_s - (12.0/H12_0) * H12_s
    dP_dc8 = H8_s - (1680.0/H12_0) * H12_s
    
    # dM/dc_i = x_star * exp_s * dP/dc_i
    # dB/dc_i = 2 * sqrt(2*pi) * M * (x_star * exp_s * dP/dc_i) * sign(f_star) ? 
    # Wait. M = |f|. dM/dc = sign(f) * df/dc.
    # B = sqrt(2*pi) M^2. dB/dc = 2 sqrt(2*pi) M dM/dc = 2 sqrt(2*pi) M sign(f) df/dc.
    # df/dc_i = x_star * exp_s * dP/dc_i.
    sign_f = 1.0 if f_star > 0 else -1.0
    factor = 2 * math.sqrt(2 * math.pi) * M * sign_f * x_star * exp_s
    
    grad = np.array([
        factor * dP_dc0,
        factor * dP_dc4,
        factor * dP_dc8
    ])
    
    return bound, grad

def construct():
    start_time = time.time()
    deadline = 200.0  # Safety margin
    
    # Initial guess
    x0 = np.array([-1.13, 0.04, 0.0003])
    
    # Global Search using Differential Evolution
    # Bounds: c0 in [-5, 5], c4 in [-1, 1], c8 in [-0.1, 0.1]
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    def objective_only(c):
        try:
            val, _ = _compute_bound_and_grad(c)
            return val
        except:
            return 1e10
            
    best_val = objective_only(x0)
    best_x = x0.copy()
    
    if time.time() < deadline - 60:
        try:
            res = differential_evolution(objective_only, bounds, seed=42, maxiter=50, tol=1e-6, polish=False)
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
            
    # Local Polish with L-BFGS-B using analytical gradient
    if time.time() < deadline - 10:
        try:
            res = minimize(objective_only, best_x, jac=lambda x: _compute_bound_and_grad(x)[1], 
                           method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
            
    # Final high-precision check
    if time.time() < deadline - 1:
        try:
            # One more step with tighter tol
            res = minimize(objective_only, best_x, jac=lambda x: _compute_bound_and_grad(x)[1], 
                           method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-18, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
            
    return best_x
# EVOLVE-BLOCK-END
