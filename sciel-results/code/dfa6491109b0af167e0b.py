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
    
    # q(x) = a0 + a1*x^2 + a2*x^4 + a3*x^6 + a4*x^8 + a5*x^10
    # Let y = x^2, then q(x) = P(y) where P(y) = a0 + a1*y + a2*y^2 + a3*y^3 + a4*y^4 + a5*y^5
    # We need real roots of q(x), which correspond to y >= 0 roots of P(y).
    # r is the largest real root of q at which q changes sign.
    
    # Find roots of P(y) = a5*y^5 + a4*y^4 + a3*y^3 + a2*y^2 + a1*y + a0
    P_coeffs = np.array([a5, a4, a3, a2, a1, a0]) if len(q_coeffs) == 6 else q_coeffs[::-1]
    # Wait, q_coeffs = [a0, a1, a2, a3, a4, a5], so P(y) coefficients in descending order:
    P_coeffs = q_coeffs[::-1]
    
    roots_y = np.roots(P_coeffs)
    
    # Filter real roots (imaginary part close to 0)
    real_roots_y = []
    for ry in roots_y:
        if abs(ry.imag) < 1e-10:
            real_roots_y.append(ry.real)
    
    # For each real root y >= 0, x = sqrt(y) and x = -sqrt(y) are roots of q.
    # We need the largest real root r of q at which q changes sign.
    # q(x) changes sign at x iff P(y) changes sign at y=x^2.
    # P(y) changes sign at a simple real root.
    
    # Find the largest x such that q changes sign at x.
    # Since q is even, roots come in pairs +/-sqrt(y).
    # The largest real root would be sqrt(max_y) where max_y is the largest non-negative real root of P.
    # But we need q to change sign there.
    
    best_r = 0.0
    for y in real_roots_y:
        if y < -1e-10:
            continue
        y = max(y, 0.0)
        x_val = np.sqrt(y)
        # Check if q changes sign at x_val
        # Evaluate q at x_val - eps and x_val + eps
        eps = 1e-8
        q_minus = (q_coeffs[0] + q_coeffs[1]*(x_val-eps)**2 + q_coeffs[2]*(x_val-eps)**4 + 
                   q_coeffs[3]*(x_val-eps)**6 + q_coeffs[4]*(x_val-eps)**8 + q_coeffs[5]*(x_val-eps)**10)
        q_plus = (q_coeffs[0] + q_coeffs[1]*(x_val+eps)**2 + q_coeffs[2]*(x_val+eps)**4 + 
                  q_coeffs[3]*(x_val+eps)**6 + q_coeffs[4]*(x_val+eps)**8 + q_coeffs[5]*(x_val+eps)**10)
        if q_minus * q_plus < 0:
            # Sign change
            if x_val > best_r:
                best_r = x_val
    
    # Also check x=0 (y=0)
    # q(0) = a0, need to check if it changes sign at 0... but 0 is a boundary.
    # Actually, if y=0 is a root, x=0 is a root. Check sign change.
    if abs(q_coeffs[0]) < 1e-10:
        # y=0 is a root
        q_plus = q_coeffs[0] + q_coeffs[1]*1e-4 + q_coeffs[2]*1e-8 + q_coeffs[3]*1e-12 + q_coeffs[4]*1e-16 + q_coeffs[5]*1e-20
        if q_plus * 0 < 0:  # This doesn't work well
            pass
        # Actually for sign change at x=0, we need q(x) to have opposite signs for x>0 small and x<0 small.
        # But q is even, so q(x)=q(-x), so it can't change sign at 0 unless it's identically 0 near 0.
        # So skip x=0 for sign change purposes (unless we consider one-sided, but the problem says "changes sign").
    
    if best_r == 0.0:
        # No sign-changing root found, return a large penalty
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
        
        if result.success or result.fun < best_val:
            best_val = result.fun
            best_coeffs = result.x
        
        elapsed = time.time() - start_time
        remaining = deadline - (start_time + elapsed)
        
        if remaining > 10:
            # Try more iterations or different starting points
            # Local refinement around the best found
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
    if time.time() > deadline - 5:
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
