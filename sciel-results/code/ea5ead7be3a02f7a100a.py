"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 145.0
    
    evals = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def call_f(x):
        nonlocal evals, best_x, best_abs
        if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = f(x)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return None
        if math.isfinite(val) and abs(val) < best_abs:
            best_abs = abs(val)
            best_x = x
        return val
    
    def iq_step(x0, f0, x1, f1, x2, f2):
        """Inverse quadratic interpolation step. Returns new x or None."""
        # Work in inverse space: g(x) = x/f(x)
        # Fit parabola through (g0, x0), (g1, x1), (g2, x2) where gi = xi/fi
        # But we want to find where f=0, so interpolate x as function of f
        # Actually, IQI: interpolate x(f) using points (f_i, x_i)
        # Newton form: x = x0 + (f-f0)*a + (f-f0)*(f-f1)*b
        # Solve for f such that... we want f_new = 0
        # Simpler: use the standard IQI formula
        
        # Standard IQI: given (x0,f0), (x1,f1), (x2,f2), find x where f=0
        # using quadratic interpolation in inverse space
        
        # Check for degenerate cases
        if f0 == 0:
            return x0
        if f1 == 0:
            return x1
        if f2 == 0:
            return x2
        
        # Use Lagrange interpolation on x as function of f
        # x(f) = x0 * L0(f) + x1 * L1(f) + x2 * L2(f)
        # where Li(f) are Lagrange basis polynomials
        # We evaluate at f=0
        
        # L0(0) = (0 - f1)(0 - f2) / ((f0 - f1)(f0 - f2))
        # L1(0) = (0 - f0)(0 - f2) / ((f1 - f0)(f1 - f2))
        # L2(0) = (0 - f0)(0 - f1) / ((f2 - f0)(f2 - f1))
        
        denom0 = (f0 - f1) * (f0 - f2)
        denom1 = (f1 - f0) * (f1 - f2)
        denom2 = (f2 - f0) * (f2 - f1)
        
        if abs(denom0) < 1e-300 or abs(denom1) < 1e-300 or abs(denom2) < 1e-300:
            return None
        
        L0 = (f1 * f2) / denom0
        L1 = (f0 * f2) / denom1
        L2 = (f0 * f1) / denom2
        
        x_new = x0 * L0 + x1 * L1 + x2 * L2
        
        if not math.isfinite(x_new):
            return None
        
        return x_new
    
    def safeguarded_iqi(x0, max_iters=30):
        """Safeguarded inverse quadratic interpolation.
        
        Uses three points and falls back to secant or bisection if needed.
        """
        if time.monotonic() > DEADLINE:
            return None
        
        x = float(x0)
        h = 1e-4 * max(1.0, abs(x))
        
        # Get three initial points
        x_a = x
        x_b = x + h
        x_c = x - h
        
        f_a = call_f(x_a)
        if f_a is None:
            return None
        if abs(f_a) <= TOL:
            return x_a
        
        f_b = call_f(x_b)
        if f_b is None:
            return None
        if abs(f_b) <= TOL:
            return x_b
        
        f_c = call_f(x_c)
        if f_c is None:
            return None
        if abs(f_c) <= TOL:
            return x_c
        
        # Sort by |f| to pick best three points
        points = [(x_a, f_a), (x_b, f_b), (x_c, f_c)]
        points.sort(key=lambda p: abs(p[1]))
        
        # Keep the three points (they're already three)
        x0, f0 = points[0]
        x1, f1 = points[1]
        x2, f2 = points[2]
        
        # Track bracket if available
        bracket_lo = None
        bracket_hi = None
        bracket_flo = None
        bracket_fhi = None
        
        # Find bracket if possible
        for (xa, fa), (xb, fb) in [(points[0], points[1]), (points[0], points[2]), (points[1], points[2])]:
            if fa * fb < 0:
                if abs(fa) < abs(fb):
                    bracket_lo, bracket_hi = xa, xb
                    bracket_flo, bracket_fhi = fa, fb
                else:
                    bracket_lo, bracket_hi = xb, xa
                    bracket_flo, bracket_fhi = fb, fa
                break
        
        for _ in range(max_iters):
            if time.monotonic() > DEADLINE:
                return None
            if evals >= MAX_EVALS - 2:
                break
            
            # Try IQI step
            x_new = iq_step(x0, f0, x1, f1, x2, f2)
            
            # If IQI fails or step is too large, use secant
            if x_new is None or not math.isfinite(x_new):
                # Fall back to secant using two closest points
                if f1 != f0:
                    x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
                else:
                    # Bisection if bracket exists
                    if bracket_lo is not None:
                        x_new = (bracket_lo + bracket_hi) / 2.0
                    else:
                        x_new = x1 + 0.5 * h
                if not math.isfinite(x_new):
                    break
            
            # Safeguard: if step is too large, clamp
            scale = max(1.0, abs(x1))
            if abs(x_new - x1) > 10 * scale:
                x_new = x1 + 10 * scale * (1 if x_new > x1 else -1)
            
            # If bracket exists, keep within bracket
            if bracket_lo is not None and bracket_hi is not None:
                if x_new < min(bracket_lo, bracket_hi) or x_new > max(bracket_lo, bracket_hi):
                    x_new = (bracket_lo + bracket_hi) / 2.0
            
            f_new = call_f(x_new)
            if f_new is None:
                break
            if abs(f_new) <= TOL:
                return x_new
            
            # Update bracket
            if bracket_lo is not None and bracket_hi is not None:
                if bracket_flo * f_new <= 0:
                    bracket_hi = x_new
                    bracket_fhi = f_new
                else:
                    bracket_lo = x_new
                    bracket_flo = f_new
            
            # Update the three points: keep the three with smallest |f|
            all_points = [(x0, f0), (x1, f1), (x2, f2), (x_new, f_new)]
            all_points.sort(key=lambda p: abs(p[1]))
            x0, f0 = all_points[0]
            x1, f1 = all_points[1]
            x2, f2 = all_points[2]
        
        return None
    
    def multi_start_siqi(x0, n_starts=8):
        """Try SIQI from multiple starting points."""
        x0 = float(x0)
        base_scale = max(1.0, abs(x0))
        
        # Generate candidate starting points
        offsets = [
            0.0,
            1e-3 * base_scale, -1e-3 * base_scale,
            1e-2 * base_scale, -1e-2 * base_scale,
            0.1 * base_scale, -0.1 * base_scale,
            1.0 * base_scale, -1.0 * base_scale,
        ][:n_starts]
        
        # Shuffle with fixed seed for reproducibility, but keep x0 first
        random.seed(42)
        random.shuffle(offsets[1:])
        offsets = [offsets[0]] + offsets[1:]
        
        for offset in offsets:
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 10:
                break
            x_start = x0 + offset
            result = safeguarded_iqi(x_start, max_iters=25)
            if result is not None:
                return result
        
        return None
    
    # Main strategy:
    # 1. Quick check at x0
    fx0 = call_f(x0)
    if fx0 is not None and abs(fx0) <= TOL:
        return x0
    
    # 2. Multi-start SIQI
    result = multi_start_siqi(x0, n_starts=10)
    if result is not None:
        return result
    
    # 3. If we haven't found a root, try a few more random starts
    if time.monotonic() < DEADLINE and evals < MAX_EVALS - 20:
        base_scale = max(1.0, abs(x0))
        for _ in range(5):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 10:
                break
            x_try = x0 + random.uniform(-5, 5) * base_scale
            result = safeguarded_iqi(x_try, max_iters=20)
            if result is not None:
                return result
    
    # Return best found
    return best_x
# EVOLVE-BLOCK-END
