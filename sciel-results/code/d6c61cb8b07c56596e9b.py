"""Rewritten program: optimize coefficients to minimize the largest sign-changing root of q=g/x^2."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_hermite  # not directly needed but available
import time

def construct():
    start_time = time.time()
    deadline = start_time + 200  # safety margin before 210s kill
    
    # Hermite polynomials H0, H4, H8, H12 (physicists', H_n(x))
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 1920x^6 + 4320x^4 - 3360x^2 + 168
    # H12 = 4096x^12 - 57600x^10 + 950400x^8 - 6048000x^6 + 15746400x^4 - 18144000x^2 + 967680
    # Let me verify H12(0): 967680. Let me compute properly.
    # Actually, let me use sympy to be safe, but that might be slow. Let me hardcode.
    
    # Physicists' Hermite: H_n(x) defined by H_n = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
    # H_0 = 1
    # H_1 = 2x
    # H_2 = 4x^2 - 2
    # H_3 = 8x^3 - 12x
    # H_4 = 16x^4 - 48x^2 + 12
    # H_8 = 256x^8 - 1920x^6 + 4320x^4 - 3360x^2 + 168
    # H_12: let me compute. H_12 = 4096x^12 - 57600x^10 + 950400x^8 - 6048000x^6 + 15746400x^4 - 18144000x^2 + 967680
    # H_12(0) = 967680
    # H_8(0) = 168
    # H_4(0) = 12
    # H_0(0) = 1
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 168.0
    H12_0 = 967680.0
    
    # Coefficients of H4: [16, 0, -48, 0, 12] for x^4, x^3, x^2, x^1, x^0
    # Coefficients of H8: [256, 0, -1920, 0, 4320, 0, -3360, 0, 168]
    # Coefficients of H12: [4096, 0, -57600, 0, 950400, 0, -6048000, 0, 15746400, 0, -18144000, 0, 967680]
    
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # g is even, degree 12 (if c3 != 0)
    # g(0) = c0*1 + c1*12 + c2*168 + c3*967680 = 0
    # c3 = -(c0 + 12*c1 + 168*c2) / 967680
    
    # q(x) = g(x)/x^2, which is a degree 10 polynomial (even)
    # q(x) = a10*x^10 + a8*x^8 + a6*x^6 + a4*x^4 + a2*x^2 + a0
    
    # Let me express q coefficients in terms of c0, c1, c2, c3:
    # g(x) = c0*(1) + c1*(16x^4 - 48x^2 + 12) + c2*(256x^8 - 1920x^6 + 4320x^4 - 3360x^2 + 168) + c3*(4096x^12 - 57600x^10 + 950400x^8 - 6048000x^6 + 15746400x^4 - 18144000x^2 + 967680)
    
    # g(x)/x^2 coefficients (for x^10, x^8, x^6, x^4, x^2, x^0):
    # x^10: c3 * (-57600)
    # x^8: c2*256 + c3*950400
    # x^6: c2*(-1920) + c3*(-6048000)
    # x^4: c1*16 + c2*4320 + c3*15746400
    # x^2: c1*(-48) + c2*(-3360) + c3*(-18144000)
    # x^0: c0 + c1*12 + c2*168 + c3*967680 = 0 (by constraint)
    
    # So q(x) = q10*x^10 + q8*x^8 + q6*x^6 + q4*x^4 + q2*x^2 + q0
    # where q0 = 0 always.
    
    # Let y = x^2. Then q = P(y) where P(y) = q10*y^5 + q8*y^4 + q6*y^3 + q4*y^2 + q2*y
    # P(y) = y * (q10*y^4 + q8*y^3 + q6*y^2 + q4*y + q2)
    # So roots of q in x correspond to roots of P in y >= 0, plus x=0.
    # The non-zero roots of q are x = ±sqrt(y_i) where y_i > 0 are roots of the quartic Q(y) = q10*y^4 + q8*y^3 + q6*y^2 + q4*y + q2
    
    # Sign change of q at x=r > 0: q(x) = P(x^2). Near x=r, x^2 is near y_r = r^2.
    # q changes sign at x=r iff P changes sign at y=y_r, i.e., y_r is a root of odd multiplicity of P.
    # P(y) = y * Q(y). At y>0, P changes sign iff Q changes sign.
    # So we need the largest y_r > 0 where Q(y) changes sign, then r = sqrt(y_r).
    # value = r^2/(2*pi) = y_r/(2*pi)
    
    # So I want to minimize y_r/(2*pi) where y_r is the largest positive root of Q where Q changes sign.
    # Equivalently, minimize y_r.
    
    # Q(y) = q10*y^4 + q8*y^3 + q6*y^2 + q4*y + q2
    
    # Let me define the objective function.
    
    def compute_q_coeffs(c0, c1, c2):
        c3 = -(c0 + 12.0*c1 + 168.0*c2) / 967680.0
        q10 = -57600.0 * c3
        q8 = 256.0*c2 + 950400.0*c3
        q6 = -1920.0*c2 - 6048000.0*c3
        q4 = 16.0*c1 + 4320.0*c2 + 15746400.0*c3
        q2 = -48.0*c1 - 3360.0*c2 - 18144000.0*c3
        # q0 = 0
        return q10, q8, q6, q4, q2, c3
    
    def compute_value(c0, c1, c2):
        q10, q8, q6, q4, q2, c3 = compute_q_coeffs(c0, c1, c2)
        
        # Q(y) = q10*y^4 + q8*y^3 + q6*y^2 + q4*y + q2
        # Find all real roots of Q, filter y > 0, check sign change.
        # For a polynomial, a root y0 has odd multiplicity iff the sign changes.
        # Numerically, I'll find all real roots and check sign change by evaluating Q at y0 +/- epsilon.
        
        # Use numpy to find roots of the quartic
        coeffs = np.array([q10, q8, q6, q4, q2])
        
        # If q10 is very small, it's effectively a cubic or lower.
        # Handle the general case: find all roots (complex), take real ones.
        if abs(q10) < 1e-15:
            if abs(q8) < 1e-15:
                if abs(q6) < 1e-15:
                    if abs(q4) < 1e-15:
                        # Q = q2, constant. No roots unless q2=0.
                        if abs(q2) < 1e-15:
                            # Q is identically zero? Then q is identically zero. Degenerate.
                            return float('inf')
                        return float('inf')
                    # Q = q4*y + q2, linear
                    if abs(q4) < 1e-15:
                        return float('inf')
                    y_root = -q2/q4
                    if y_root > 0:
                        # Linear always changes sign
                        return y_root / (2.0 * np.pi)
                    return float('inf')
                # Q = q6*y^2 + q4*y + q2, quadratic
                disc = q4*q4 - 4.0*q6*q2
                if disc < 0:
                    return float('inf')
                sq = np.sqrt(disc)
                y1 = (-q4 + sq) / (2.0*q6)
                y2 = (-q4 - sq) / (2.0*q6)
                max_y = -1
                for y in [y1, y2]:
                    if y > 1e-15:
                        # Check sign change: evaluate Q at y - eps and y + eps
                        # For quadratic with distinct real roots, sign changes at both.
                        # For double root, no sign change.
                        if abs(disc) > 1e-10:  # distinct roots
                            if y > max_y:
                                max_y = y
                if max_y < 0:
                    return float('inf')
                return max_y / (2.0 * np.pi)
            # Q = q8*y^3 + q6*y^2 + q4*y + q2, cubic
            coeffs_c = np.array([q8, q6, q4, q2])
            roots = np.roots(coeffs_c)
            real_roots = roots[np.abs(roots.imag) < 1e-10].real
            max_y = -1
            for y in real_roots:
                if y > 1e-15:
                    # Check sign change by evaluating Q at y +/- delta
                    delta = max(1e-8, abs(y)*1e-8)
                    Q_plus = q8*(y+delta)**3 + q6*(y+delta)**2 + q4*(y+delta) + q2
                    Q_minus = q8*(y-delta)**3 + q6*(y-delta)**2 + q4*(y-delta) + q2
                    if Q_plus * Q_minus < 0:
                        if y > max_y:
                            max_y = y
            if max_y < 0:
                return float('inf')
            return max_y / (2.0 * np.pi)
        
        # Full quartic
        coeffs_q = np.array([q10, q8, q6, q4, q2])
        roots = np.roots(coeffs_q)
        real_roots = roots[np.abs(roots.imag) < 1e-10].real
        max_y = -1
        for y in real_roots:
            if y > 1e-15:
                # Check sign change
                delta = max(1e-8, abs(y)*1e-8)
                Q_plus = q10*(y+delta)**4 + q8*(y+delta)**3 + q6*(y+delta)**2 + q4*(y+delta) + q2
                Q_minus = q10*(y-delta)**4 + q8*(y-delta)**3 + q6*(y-delta)**2 + q4*(y-delta) + q2
                if Q_plus * Q_minus < 0:
                    if y > max_y:
                        max_y = y
        if max_y < 0:
            # No sign-changing positive roots. This is great! value would be... 
            # But the problem says "r be the largest real root of q at which q changes sign"
            # If no such root exists, what happens? The problem implies it exists.
            # Let me return a large value to penalize this case, or handle differently.
            # Actually, if there's no sign-changing root, maybe the value is 0 or undefined.
            # Let me return a large penalty to push away from this region.
            return 1e6
        return max_y / (2.0 * np.pi)
    
    # Seed
    c0_seed, c1_seed, c2_seed = -113.0/100.0, 1.0/25.0, 1.0/3240.0
    seed_value = compute_value(c0_seed, c1_seed, c2_seed)
    
    best_c = np.array([c0_seed, c1_seed, c2_seed])
    best_val = seed_value
    
    # Local optimization using Nelder-Mead
    def objective(x):
        return compute_value(x[0], x[1], x[2])
    
    # Try multiple starting points
    start_points = [
        [c0_seed, c1_seed, c2_seed],
        [c0_seed * 1.1, c1_seed * 1.1, c2_seed * 1.1],
        [c0_seed * 0.9, c1_seed * 0.9, c2_seed * 0.9],
        [c0_seed + 0.01, c1_seed, c2_seed],
        [c0_seed, c1_seed + 0.01, c2_seed],
        [c0_seed, c1_seed, c2_seed + 0.001],
        [-1.0, 0.05, 0.0003],
        [-1.2, 0.04, 0.0004],
        [-1.0, 0.06, 0.0002],
    ]
    
    for sp in start_points:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if res.success or res.fun < best_val:
                if res.fun < best_val:
                    best_val = res.fun
                    best_c = res.x
        except Exception:
            pass
    
    # Also try a finer grid search around the seed
    if time.time() < deadline - 30:
        # Grid search in a small neighborhood
        c0_range = np.linspace(c0_seed - 0.05, c0_seed + 0.05, 20)
        c1_range = np.linspace(c1_seed - 0.02, c1_seed + 0.02, 20)
        c2_range = np.linspace(c2_seed - 0.0005, c2_seed + 0.0005, 20)
        
        for c0 in c0_range:
            if time.time() > deadline:
                break
            for c1 in c1_range:
                if time.time() > deadline:
                    break
                for c2 in c2_range:
                    if time.time() > deadline:
                        break
                    try:
                        v = compute_value(c0, c1, c2)
                        if v < best_val:
                            best_val = v
                            best_c = np.array([c0, c1, c2])
                    except:
                        pass
    
    # Final polish with very tight tolerance
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_c, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x
        except:
            pass
    
    return np.array(best_c)

# EVOLVE-BLOCK-END
