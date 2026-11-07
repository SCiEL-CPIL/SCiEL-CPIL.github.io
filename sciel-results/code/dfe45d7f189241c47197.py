"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import sympy as sp
from sympy.polys.orthopolys import hermite_poly
import mpmath as mp
from fractions import Fraction

# Precompute Hermite polynomial values at 0 (exact rationals as floats)
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 168.0
H12_0 = 18480.0

def compute_q_y_coeffs_exact(c0, c1, c2):
    """Compute coefficients of q(y) where y=x^2, degree 5, using exact rational arithmetic.
    Returns list [a0, a1, a2, a3, a4, a5] as Fractions for q(y) = a0 + a1*y + ... + a5*y^5."""
    c0_frac = Fraction(c0).limit_denominator(10**12)
    c1_frac = Fraction(c1).limit_denominator(10**12)
    c2_frac = Fraction(c2).limit_denominator(10**12)
    
    c3_frac = -(c0_frac * Fraction(1) + c1_frac * Fraction(12) + c2_frac * Fraction(168)) / Fraction(18480)
    
    # q(y) = (c0*H0 + c1*H4 + c2*H8 + c3*H12) / x^2
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
    # H12 = 4096x^12 - 46080x^10 + 332160x^8 - 1003520x^6 + 1424640x^4 - 921600x^2 + 18480
    
    # Divide each by x^2:
    # H0/x^2 = 1/x^2 -> this doesn't divide exactly! Wait...
    
    # Actually, g(x) = c0*H0(x) + c1*H4(x) + c2*H8(x) + c3*H12(x)
    # g(0) = 0 means c0 + 12*c1 + 168*c2 + 18480*c3 = 0
    # 
    # g(x)/x^2: H0(x) = 1, so c0*1/x^2 is not a polynomial.
    # But wait - the task says "Let q = g / x^2, which must divide exactly."
    # This means g(x) must be divisible by x^2, which requires g(0) = 0 AND g'(0) = 0.
    # 
    # g'(x) = c0*H0'(x) + c1*H4'(x) + c2*H8'(x) + c3*H12'(x)
    # H0'(x) = 0, H4'(0) = 0 (since H4 is even), etc.
    # All Hermite polynomials are even or odd. H0 is even (constant), so H0' = 0.
    # H4 is even, so H4'(0) = 0. H8 is even, so H8'(0) = 0. H12 is even, so H12'(0) = 0.
    # So g'(0) = 0 automatically for any c0, c1, c2, c3.
    # 
    # But g(0) = c0 + 12*c1 + 168*c2 + 18480*c3 = 0 is our constraint.
    # For g(x)/x^2 to be a polynomial, we need g(x) to have a factor of x^2.
    # Since all H_{2k}(x) are even polynomials, g(x) is even.
    # g(0) = 0 means g(x) = x^2 * h(x) for some polynomial h.
    # Since g is even, h is also even, so h(x) = q(x^2) for some q(y).
    # 
    # Let's compute q(y) by dividing g(x) by x^2 and substituting y = x^2.
    
    # g(x) = c0*1 + c1*(16x^4 - 48x^2 + 12) + c2*(256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168) 
    #        + c3*(4096x^12 - 46080x^10 + 332160x^8 - 1003520x^6 + 1424640x^4 - 921600x^2 + 18480)
    
    # g(x)/x^2 = c0/x^2 + c1*(16x^2 - 48 + 12/x^2) + c2*(256x^6 - 1344x^4 + 2688x^2 - 1344 + 168/x^2)
    #             + c3*(4096x^10 - 46080x^8 + 332160x^6 - 1003520x^4 + 1424640x^2 - 921600 + 18480/x^2)
    
    # The 1/x^2 terms: (c0 + 12*c1 + 168*c2 + 18480*c3)/x^2 = 0 by our constraint.
    
    # So q(y) where y = x^2:
    # q(y) = c1*(16y - 48) + c2*(256y^3 - 1344y^2 + 2688y - 1344) + c3*(4096y^5 - 46080y^4 + 332160y^3 - 1003520y^2 + 1424640y - 921600)
    
    a0 = c1_frac * Fraction(-48) + c2_frac * Fraction(-1344) + c3_frac * Fraction(-921600)
    a1 = c1_frac * Fraction(16) + c2_frac * Fraction(2688) + c3_frac * Fraction(1424640)
    a2 = c2_frac * Fraction(-1344) + c3_frac * Fraction(-1003520)
    a3 = c2_frac * Fraction(256) + c3_frac * Fraction(332160)
    a4 = c3_frac * Fraction(-46080)
    a5 = c3_frac * Fraction(4096)
    
    return [a0, a1, a2, a3, a4, a5]

def evaluate_value_precise(c0, c1, c2):
    """Compute r^2/(2*pi) with high precision."""
    coeffs = compute_q_y_coeffs_exact(c0, c1, c2)
    
    c3_frac = -(Fraction(c0).limit_denominator(10**12) * Fraction(1) + 
                 Fraction(c1).limit_denominator(10**12) * Fraction(12) + 
                 Fraction(c2).limit_denominator(10**12) * Fraction(168)) / Fraction(18480)
    
    # If c3 < 0, g -> -inf at infinity, so replace g by -g
    if c3_frac < 0:
        coeffs = [-c for c in coeffs]
        c3_frac = -c3_frac
    
    if c3_frac <= 0:
        return 1e10
    
    # Convert to mpmath for high-precision root finding
    mp.dps = 50
    a0_mp = mp.mpf(coeffs[0])
    a1_mp = mp.mpf(coeffs[1])
    a2_mp = mp.mpf(coeffs[2])
    a3_mp = mp.mpf(coeffs[3])
    a4_mp = mp.mpf(coeffs[4])
    a5_mp = mp.mpf(coeffs[5])
    
    # q(y) = a5*y^5 + a4*y^4 + a3*y^3 + a2*y^2 + a1*y + a0
    def q_func(y):
        return a5_mp * y**5 + a4_mp * y**4 + a3_mp * y**3 + a2_mp * y**2 + a1_mp * y + a0_mp
    
    # Find all positive real roots using numpy for initial estimates
    coeffs_np = np.array([a5_mp, a4_mp, a3_mp, a2_mp, a1_mp, a0_mp], dtype=np.float64)
    roots_np = np.roots(coeffs_np)
    real_roots_np = roots_np[np.abs(roots_np.imag) < 1e-10].real
    positive_roots_np = real_roots_np[real_roots_np > 1e-10]
    
    if len(positive_roots_np) == 0:
        return 1e10
    
    # Refine each root with mpmath findroot and check sign change
    sign_change_roots = []
    for r_np in positive_roots_np:
        r_mp = mp.mpf(str(r_np))
        # Try to refine with Newton's method
        try:
            def dq_func(y):
                return 5*a5_mp*y**4 + 4*a4_mp*y**3 + 3*a3_mp*y**2 + 2*a2_mp*y + a1_mp
            
            # Newton refinement
            y_cur = r_mp
            for _ in range(20):
                q_val = q_func(y_cur)
                dq_val = dq_func(y_cur)
                if abs(dq_val) < mp.mpf('1e-40'):
                    break
                y_new = y_cur - q_val / dq_val
                if abs(y_new - y_cur) < mp.mpf('1e-40'):
                    y_cur = y_new
                    break
                y_cur = y_new
            
            # Check if it's actually a root (close to zero)
            if abs(q_func(y_cur)) < mp.mpf('1e-30'):
                # Check sign change
                eps = mp.mpf('1e-20') * max(mp.mpf(1), abs(y_cur))
                q_left = q_func(y_cur - eps)
                q_right = q_func(y_cur + eps)
                if q_left * q_right < 0:
                    sign_change_roots.append(y_cur)
        except Exception:
            # Fallback: just use the numpy root if it looks reasonable
            if q_func(mp.mpf(str(r_np))) is not None and abs(q_func(mp.mpf(str(r_np)))) < mp.mpf('1e-10'):
                sign_change_roots.append(mp.mpf(str(r_np)))
    
    if len(sign_change_roots) == 0:
        return 1e10
    
    y_max = max(sign_change_roots)
    result = float(y_max / (2 * mp.pi))
    return result

def objective(params):
    """Objective: minimize the value."""
    try:
        val = evaluate_value_precise(params[0], params[1], params[2])
        if np.isnan(val) or np.isinf(val):
            return 1e10
        return val
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 190  # Safety margin before 210s kill
    
    # Seed
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best = seed.copy()
    best_val = evaluate_value_precise(*seed)
    
    # Phase 1: Coarse grid search to identify promising regions
    elapsed = time.time() - start_time
    if elapsed < deadline - 120:
        # Grid around the seed
        c0_range = np.linspace(-2.0, 0.0, 10)
        c1_range = np.linspace(-0.5, 1.0, 8)
        c2_range = np.linspace(-0.3, 0.3, 8)
        
        for c0 in c0_range:
            for c1 in c1_range:
                if time.time() > deadline - 120:
                    break
                for c2 in c2_range:
                    try:
                        val = evaluate_value_precise(c0, c1, c2)
                        if val < best_val and val < 1e9:
                            best = np.array([c0, c1, c2])
                            best_val = val
                    except Exception:
                        pass
    
    # Phase 2: Local optimization from best and several nearby points
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        # Refine the best point
        try:
            nm_result = minimize(
                objective, best,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if nm_result.fun < best_val:
                best = nm_result.x
                best_val = nm_result.fun
        except Exception:
            pass
    
    # Phase 3: Multiple local optimizations from diverse starting points
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        rng = np.random.RandomState(42)
        # Start points: around seed, around current best, and random
        start_points = [best.copy(), seed.copy()]
        for i in range(10):
            if time.time() > deadline - 30:
                break
            # Diverse perturbations
            scale = 0.05 + 0.3 * rng.rand()
            x0 = best + rng.normal(0, scale, 3)
            start_points.append(x0)
        
        for x0 in start_points:
            if time.time() > deadline - 30:
                break
            try:
                local_result = minimize(
                    objective, x0,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True}
                )
                if local_result.fun < best_val:
                    best = local_result.x
                    best_val = local_result.fun
            except Exception:
                pass
    
    # Phase 4: Final ultra-fine refinement
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            final_result = minimize(
                objective, best,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            if final_result.fun < best_val:
                best = final_result.x
                best_val = final_result.fun
        except Exception:
            pass
    
    # Phase 5: One more refinement pass with even tighter tolerance
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        try:
            final2_result = minimize(
                objective, best,
                method='Powell',
                options={'maxiter': 2000, 'xtol': 1e-14, 'ftol': 1e-14}
            )
            if final2_result.fun < best_val:
                best = final2_result.x
                best_val = final2_result.fun
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
