"""Combined search for the Uncertainty Inequality task."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Deadline: evaluator kills at 210s. Use 190s budget to be safe.
_BUDGET_SECONDS = 190.0

# Pre-computed Hermite polynomial coefficients (physicists' convention)
# H0(x) = 1
# H4(x) = 16*x^4 - 48*x^2 + 12
# H8(x) = 256*x^8 - 4480*x^6 + 24192*x^4 - 40320*x^2 + 1680
# H12(x) = 4096*x^12 - 147456*x^10 + 1684608*x^8 - 9067200*x^6 + 20901120*x^4 - 20321280*x^2 + 665280

# Values at x=0
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 665280.0

def _compute_value(c0, c1, c2):
    """
    Compute the value = r^2 / (2*pi) for the given coefficients.
    Returns a positive float (lower is better), or inf if invalid.
    """
    try:
        # c3 is fixed so that g(0) = 0
        c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
        
        # Coefficients for g(x) = a0 + a2*x^2 + a4*x^4 + a6*x^6 + a8*x^8 + a10*x^10 + a12*x^12
        a0 = c0 + c1*_H4_0 + c2*_H8_0 + c3*_H12_0
        # a0 should be 0 by construction
        
        a2 = c1*(-48) + c2*(-40320) + c3*(-20321280)
        a4 = c1*16 + c2*24192 + c3*20901120
        a6 = c2*(-4480) + c3*(-9067200)
        a8 = c2*256 + c3*1684608
        a10 = c3*(-147456)
        a12 = c3*4096
        
        # Check if g tends to a negative limit at infinity.
        # g(x) ~ c3*4096*x^12 as x -> infinity.
        # If c3 < 0, then g -> -infinity, so we replace g by -g, which flips all coefficients.
        if c3 < 0:
            a2 = -a2
            a4 = -a4
            a6 = -a6
            a8 = -a8
            a10 = -a10
            a12 = -a12
        
        # q(x) = g(x) / x^2. Since g(0)=0 and all terms have even powers, this divides exactly.
        # q(x) = a2 + a4*x^2 + a6*x^4 + a8*x^6 + a10*x^8 + a12*x^10
        # Let y = x^2, then q = P(y) where P(y) = a2 + a4*y + a6*y^2 + a8*y^3 + a10*y^4 + a12*y^5
        
        # We need the largest real root of q at which q changes sign.
        # This means the largest y > 0 such that P(y) = 0 and P changes sign at y.
        # Then r = sqrt(y), and value = r^2 / (2*pi) = y / (2*pi).
        
        # Coefficients for P(y) in ascending order: [a2, a4, a6, a8, a10, a12]
        coeffs_P = np.array([a2, a4, a6, a8, a10, a12])
        
        # Find real positive roots
        roots = np.roots(coeffs_P)
        real_roots = []
        for root in roots:
            if abs(root.imag) < 1e-10:
                real_root = root.real
                if real_root > 1e-12:
                    real_roots.append(real_root)
        
        if not real_roots:
            return float('inf')
        
        # Sort real positive roots
        real_roots.sort()
        
        # For each root, check if P changes sign at that root.
        # P changes sign at y if P(y - eps) and P(y + eps) have opposite signs.
        best_y = None
        for y in real_roots:
            eps = 1e-8 * max(1.0, abs(y))
            # Evaluate P at y-eps and y+eps
            # P(y) = a2 + a4*y + a6*y^2 + a8*y^3 + a10*y^4 + a12*y^5
            # Using np.polyval with coefficients in descending order: [a12, a10, a8, a6, a4, a2]
            P_minus = np.polyval(coeffs_P[::-1], y - eps)
            P_plus = np.polyval(coeffs_P[::-1], y + eps)
            if P_minus * P_plus < 0:
                if best_y is None or y > best_y:
                    best_y = y
        
        if best_y is None:
            return float('inf')
        
        value = best_y / (2 * math.pi)
        return value
    
    except Exception:
        return float('inf')


def _seed_value():
    """Return the value for the seed coefficients."""
    return _compute_value(-113.0/100.0, 1.0/25.0, 1.0/3240.0)


def construct():
    start_time = time.time()
    
    # Start with the seed
    best_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_value = _compute_value(best_c[0], best_c[1], best_c[2])
    
    def _time_left():
        return _BUDGET_SECONDS - (time.time() - start_time)
    
    # Strategy 1: Local optimization from seed using Nelder-Mead
    if _time_left() > 30:
        try:
            res = minimize(
                lambda x: _compute_value(x[0], x[1], x[2]),
                best_c,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res.fun < best_value:
                best_value = res.fun
                best_c = res.x
        except Exception:
            pass
    
    # Strategy 2: Differential evolution over a reasonable range
    if _time_left() > 60:
        try:
            bounds = [
                (-5.0, 5.0),
                (-5.0, 5.0),
                (-0.5, 0.5)
            ]
            res = differential_evolution(
                lambda x: _compute_value(x[0], x[1], x[2]),
                bounds,
                maxiter=100,
                popsize=30,
                tol=1e-10,
                seed=42,
                init='sobol'
            )
            if res.fun < best_value:
                best_value = res.fun
                best_c = res.x
        except Exception:
            pass
    
    # Strategy 3: Multi-start local optimization from random points
    if _time_left() > 60:
        rng = np.random.default_rng(12345)
        n_starts = 0
        while _time_left() > 20:
            x0 = rng.uniform(-5.0, 5.0, size=3)
            x0[2] *= 0.1  # c2 is typically small
            try:
                res = minimize(
                    lambda x: _compute_value(x[0], x[1], x[2]),
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-15}
                )
                if res.fun < best_value:
                    best_value = res.fun
                    best_c = res.x
            except Exception:
                pass
            n_starts += 1
            if n_starts > 200:
                break
    
    # Strategy 4: Final high-precision polish pass
    if _time_left() > 10:
        try:
            res = minimize(
                lambda x: _compute_value(x[0], x[1], x[2]),
                best_c,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-16, 'fatol': 1e-18}
            )
            if res.fun < best_value:
                best_value = res.fun
                best_c = res.x
        except Exception:
            pass
    
    return np.array(best_c, dtype=np.float64)

# EVOLVE-BLOCK-END
