"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Precompute Hermite polynomial coefficients at needed points
# H_n(x) physicists' Hermite: H0=1, H4=16x^4-48x^2+12, H8=256x^8-1920x^6+3360x^4-1680x^2+168
# H12 = 4096x^12 - 46080x^10 + 230400x^8 - 403200x^6 + 302400x^4 - 84000x^2 + 5040

def H0(x):
    return 1.0

def H4(x):
    return 16*x**4 - 48*x**2 + 12

def H8(x):
    return 256*x**8 - 1920*x**6 + 3360*x**4 - 1680*x**2 + 168

def H12(x):
    return 4096*x**12 - 46080*x**10 + 230400*x**8 - 403200*x**6 + 302400*x**4 - 84000*x**2 + 5040

# Values at x=0
H0_0 = H0(0)  # 1
H4_0 = H4(0)  # 12
H8_0 = H8(0)  # 168
H12_0 = H12(0)  # 5040

def compute_q_coefficients(c0, c1, c2):
    """
    Given c0, c1, c2, compute c3 so that g(0)=0, then form q(x) = g(x)/x^2.
    g(x) = c0*H0(x) + c1*H4(x) + c2*H8(x) + c3*H12(x)
    q(x) = g(x)/x^2, which must divide exactly (no remainder).
    """
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g(x) = c0*(1) + c1*(16x^4 - 48x^2 + 12) + c2*(256x^8 - 1920x^6 + 3360x^4 - 1680x^2 + 168) + c3*(4096x^12 - 46080x^10 + 230400x^8 - 403200x^6 + 302400x^4 - 84000x^2 + 5040)
    
    # Collect coefficients of x^k in g(x):
    # x^0: c0 + 12*c1 + 168*c2 + 5040*c3 = 0 (by construction)
    # x^2: -48*c1 - 1680*c2 - 84000*c3
    # x^4: 16*c1 + 3360*c2 + 302400*c3
    # x^6: -1920*c2 - 403200*c3
    # x^8: 256*c2 + 230400*c3
    # x^10: -46080*c3
    # x^12: 4096*c3
    
    # q(x) = g(x)/x^2, so:
    # q(x) = a0 + a1*x^2 + a2*x^4 + a3*x^6 + a4*x^8 + a5*x^10 + a6*x^12
    # where:
    # a0 = -48*c1 - 1680*c2 - 84000*c3
    # a1 = 16*c1 + 3360*c2 + 302400*c3
    # a2 = -1920*c2 - 403200*c3
    # a3 = 256*c2 + 230400*c3
    # a4 = -46080*c3
    # a5 = 0 (no x^10 term in g... wait, g has x^10 term -46080*c3, so q has x^8 term)
    
    # Let me redo: g(x) terms:
    # x^0: c0 + 12c1 + 168c2 + 5040c3 = 0
    # x^2: -48c1 - 1680c2 - 84000c3
    # x^4: 16c1 + 3360c2 + 302400c3
    # x^6: -1920c2 - 403200c3
    # x^8: 256c2 + 230400c3
    # x^10: -46080c3
    # x^12: 4096c3
    
    # q(x) = g(x)/x^2:
    # x^0: -48c1 - 1680c2 - 84000c3
    # x^2: 16c1 + 3360c2 + 302400c3
    # x^4: -1920c2 - 403200c3
    # x^6: 256c2 + 230400c3
    # x^8: -46080c3
    # x^10: 4096c3
    
    a0 = -48*c1 - 1680*c2 - 84000*c3
    a1 = 16*c1 + 3360*c2 + 302400*c3
    a2 = -1920*c2 - 403200*c3
    a3 = 256*c2 + 230400*c3
    a4 = -46080*c3
    a5 = 4096*c3
    
    return np.array([a0, a1, a2, a3, a4, a5])


def objective(coeffs):
    """
    Compute the value = r^2 / (2*pi) where r is the largest real root of q at which q changes sign.
    Lower is better.
    """
    c0, c1, c2 = coeffs
    q_coeffs = compute_q_coefficients(c0, c1, c2)
    
    # q_coeffs = [a0, a1, a2, a3, a4, a5]
    # q(x) = a0 + a1*x^2 + a2*x^4 + a3*x^6 + a4*x^8 + a5*x^10
    # Let y = x^2, P(y) = a0 + a1*y + a2*y^2 + a3*y^3 + a4*y^4 + a5*y^5
    # Coefficients of P(y) in descending power order for np.roots: [a5, a4, a3, a2, a1, a0]
    
    # Handle case where leading coefficient is 0 to avoid issues with np.roots
    P_coeffs = q_coeffs[::-1]
    
    # Remove leading zeros if any
    while len(P_coeffs) > 1 and abs(P_coeffs[0]) < 1e-12:
        P_coeffs = P_coeffs[1:]
        
    if len(P_coeffs) == 0:
        return 1e10
        
    try:
        roots_y = np.roots(P_coeffs)
    except Exception:
        return 1e10

    best_r = 0.0
    for ry in roots_y:
        if abs(ry.imag) > 1e-9:
            continue
        y = ry.real
        if y < -1e-9:
            continue
        y = max(0.0, y)
        x_val = np.sqrt(y)
        
        # Check sign change of q(x) at x_val
        # q(x) = P(x^2)
        # We evaluate P(y) at y-eps and y+eps
        eps = 1e-7
        y_minus = max(0.0, y - eps)
        y_plus = y + eps
        
        # Evaluate P(y) using Horner's method or polynomial eval
        # P(y) = sum_{i=0}^5 q_coeffs[i] * y^i
        p_minus = 0.0
        p_plus = 0.0
        for i in range(len(q_coeffs)):
            p_minus += q_coeffs[i] * (y_minus ** i)
            p_plus += q_coeffs[i] * (y_plus ** i)
            
        # If y=0, we only check one side? No, "changes sign" implies crossing.
        # If y=0 is a root, q(0)=0. Does it change sign?
        # q(x) is even. q(x) ~ a1*x^2 for small x if a0=0.
        # If a0=0 and a1 != 0, q(x) has same sign on both sides of 0 (parabola vertex). No sign change.
        # If a0=0 and a1=0, q(x) ~ a2*x^4. Same sign.
        # Generally, even functions do not change sign at 0 unless they are identically 0 in a neighborhood or oscillate, 
        # but polynomials are analytic. A non-zero even polynomial cannot change sign at 0.
        # So we only care about x > 0.
        
        if y > 1e-9:
             if p_minus * p_plus < 0:
                 if x_val > best_r:
                     best_r = x_val
        else:
             # y is very close to 0. Check if it's a root and if sign changes for x>0?
             # As established, even polynomials don't change sign at 0.
             pass

    if best_r == 0.0:
        return 1e10
    
    value = best_r**2 / (2 * np.pi)
    return value


def construct():
    """
    Find three coefficients [c0, c1, c2] that minimize the value.
    """
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 210s limit
    
    # Seed
    seed = np.array([-113/100, 1/25, 1/3240])
    best_val = objective(seed)
    best_coeffs = seed.copy()
    
    # Nelder-Mead optimization
    try:
        result = minimize(
            objective,
            seed,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        if result.fun < best_val:
            best_val = result.fun
            best_coeffs = result.x
        
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 20:
        return best_coeffs

    # Try local refinement
    try:
        result2 = minimize(
            objective,
            best_coeffs,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if result2.fun < best_val:
            best_val = result2.fun
            best_coeffs = result2.x
    except Exception:
        pass

    # Check deadline
    if time.time() > deadline - 10:
        return best_coeffs

    # Try a few different initial points to escape local minima
    np.random.seed(42)
    for i in range(20):
        if time.time() > deadline - 10:
            break
        # Perturb the best solution
        perturbation = np.random.randn(3) * 0.1
        init = best_coeffs + perturbation
        try:
            res = minimize(
                objective,
                init,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except Exception:
            pass
    
    return best_coeffs

# EVOLVE-BLOCK-END
