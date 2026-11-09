import time
import math

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find root x of f(x) such that |f(x)| <= 1e-8.
    Strategy:
    1. Evaluate f(x0). If good, return.
    2. Try a few small perturbations to find a second point (or a better first point).
    3. Use Secant method for fast local convergence (superlinear).
    4. If Secant fails to converge quickly or diverges, switch to a bracketing
       hybrid (Bisection + Inverse Quadratic Interpolation) to guarantee convergence
       if a sign change is found.
    5. If no bracket is found, return the best point encountered.
    """
    start_time = time.time()
    # The evaluator kills at 150s. We have 3 equations.
    # 10s per equation is safe. 5s is even safer and likely sufficient for <500 calls.
    deadline = start_time + 8.0
    
    tol_final = 1e-8
    tol_check = 1e-8  # Match the requirement to allow early exit
    
    x0 = float(x0)
    
    # --- Phase 0: Initial Evaluation and Perturbation ---
    f0 = f(x0)
    if not math.isfinite(f0):
        # x0 is bad. Try to find a good starting point nearby.
        # This is unlikely given the problem description, but good for robustness.
        best_x = x0
        best_f = float('inf')
        base_step = 1e-4 * max(1.0, abs(x0))
        found_finite = False
        for s in [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64, 128, -128, 256, -256, 512, -512, 1024, -1024]:
            if time.time() > deadline:
                return best_x if math.isfinite(best_f) else x0
            x_try = x0 + s * base_step
            f_try = f(x_try)
            if math.isfinite(f_try):
                if not found_finite:
                    x0, f0 = x_try, f_try
                    found_finite = True
                    break
                if abs(f_try) < abs(best_f):
                    best_x, best_f = x_try, f_try
            if abs(f_try) <= tol_check:
                return x_try
        if not found_finite:
            return x0
            
    if abs(f0) <= tol_final:
        return x0

    # We need a second point for Secant.
    # Use a small step.
    base_step = 1e-5 * max(1.0, abs(x0))
    
    x_prev = x0
    f_prev = f0
    x_curr = x0 + base_step
    f_curr = f(x_curr)
    
    if abs(f_curr) <= tol_final:
        return x_curr
    
    # Handle non-finite f_curr
    if not math.isfinite(f_curr):
        x_curr = x0 - base_step
        f_curr = f(x_curr)
        if not math.isfinite(f_curr):
            x_curr = x0 + base_step * 10
            f_curr = f(x_curr)
            if not math.isfinite(f_curr):
                # If we can't get a second finite point, we're stuck with one.
                # Return best we have.
                return x0

    best_x = x_prev if abs(f_prev) < abs(f_curr) else x_curr
    best_f = min(abs(f_prev), abs(f_curr))

    # --- Phase 1: Secant Method ---
    # Secant is very fast (order ~1.6) for smooth functions.
    # We limit iterations to avoid wasting time if it fails.
    max_secant_iter = 25
    
    for _ in range(max_secant_iter):
        if time.time() > deadline:
            break
            
        denom = f_curr - f_prev
        if abs(denom) < 1e-15:
            # Slope is zero or points are identical.
            # Try a small perturbation to restart secant?
            # Or break to bracketing phase.
            break
            
        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        
        if not math.isfinite(x_next):
            break
            
        f_next = f(x_next)
        
        if not math.isfinite(f_next):
            break
            
        if abs(f_next) < best_f:
            best_x = x_next
            best_f = abs(f_next)
            
        if abs(f_next) <= tol_final:
            return x_next
            
        # Check for oscillation/divergence: if |x_next| is getting huge, stop secant
        if abs(x_next) > 1e10 * max(1.0, abs(x0)):
            break

        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_next, f_next
        
    # Check if secant converged
    if abs(f_curr) <= tol_final:
        return x_curr
    if abs(f_prev) <= tol_final:
        return x_prev

    # --- Phase 2: Bracketing Search ---
    # We have two points (x_prev, f_prev) and (x_curr, f_curr).
    # If they have opposite signs, we have a bracket.
    # If not, we need to expand the search range to find a sign change.
    
    a, fa = x_prev, f_prev
    b, fb = x_curr, f_curr
    
    # Ensure a < b for consistent logic, though sign check doesn't strictly need order
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        
    if fa * fb < 0:
        bracket_found = True
    else:
        bracket_found = False
        # Expand from the point with smaller |f|
        if abs(fa) < abs(fb):
            # Expand from a
            # Try steps in both directions? No, a and b are already on one side.
            # We should try to jump over the root.
            # Heuristic: if f is oscillatory, the period might be small.
            # Try expanding b further in the direction away from a, and also check behind a.
            
            # Let's try a few expansions
            step_size = max(abs(b - a), 1e-4 * max(1.0, abs(a)))
            
            # Try expanding b
            for i in range(10):
                if time.time() > deadline:
                    break
                b_new = b + step_size * (1 + i)
                fb_new = f(b_new)
                if math.isfinite(fb_new):
                    if fa * fb_new < 0:
                        b, fb = b_new, fb_new
                        bracket_found = True
                        break
                    if abs(fb_new) < abs(fb):
                        # Keep the best candidate if we don't find a bracket yet?
                        # Actually, for bisection we need a bracket.
                        # Let's just keep expanding.
                        pass
                # Try expanding a in the other direction
                a_new = a - step_size * (1 + i)
                fa_new = f(a_new)
                if math.isfinite(fa_new):
                    if fa_new * fb < 0:
                        a, fa = a_new, fa_new
                        bracket_found = True
                        break

            if not bracket_found:
                # Try a wider jump
                wide_step = 1.0 * max(1.0, abs(a))
                for i in range(5):
                    if time.time() > deadline:
                        break
                    # Check points at +/- wide_step
                    x_try = a + wide_step * (i + 1)
                    f_try = f(x_try)
                    if math.isfinite(f_try):
                        if fa * f_try < 0:
                            b, fb = x_try, f_try
                            bracket_found = True
                            break
                    x_try = a - wide_step * (i + 1)
                    f_try = f(x_try)
                    if math.isfinite(f_try):
                        if fa * f_try < 0:
                            a, fa = x_try, f_try
                            bracket_found = True
                            break
                    if abs(f_try) <= tol_final:
                        return x_try

    if not bracket_found:
        # No bracket found. Return best point.
        return best_x

    # --- Phase 3: Hybrid Bisection / Inverse Quadratic Interpolation ---
    # We have a bracket [a, b] with fa*fb < 0.
    # We want to converge to |f(x)| <= 1e-8.
    # Bisection is safe but slow (log2(range/tol)).
    # IQI is faster but can fail.
    # Strategy: Use IQI if the point stays in the bracket, else bisection.
    
    # We need a third point for IQI. Use midpoint.
    m = (a + b) / 2.0
    fm = f(m)
    
    if not math.isfinite(fm):
        # Fallback to pure bisection
        for _ in range(100):
            if time.time() > deadline:
                break
            if abs(fa) <= tol_final:
                return a
            if abs(fb) <= tol_final:
                return b
            if abs(b - a) < 1e-15:
                return (a + b) / 2.0
            mid = (a + b) / 2.0
            fmid = f(mid)
            if not math.isfinite(fmid):
                break
            if abs(fmid) <= tol_final:
                return mid
            if fa * fmid < 0:
                b, fb = mid, fmid
            else:
                a, fa = mid, fmid
        return a if abs(fa) < abs(fb) else b

    if abs(fm) <= tol_final:
        return m

    # Now we have 3 points: (a, fa), (m, fm), (b, fb)
    # Sort them by x for easier management?
    # Let's keep track of x1, x2, x3 and f1, f2, f3
    # x1 < x2 < x3
    
    points = [(a, fa), (m, fm), (b, fb)]
    points.sort(key=lambda p: p[0])
    x1, f1 = points[0]
    x2, f2 = points[1]
    x3, f3 = points[2]
    
    # Determine bracket: either [x1, x2] or [x2, x3]
    if f1 * f2 < 0:
        x_lo, f_lo, x_hi, f_hi = x1, f1, x2, f2
        x_mid, f_mid = x3, f3
    elif f2 * f3 < 0:
        x_lo, f_lo, x_hi, f_hi = x2, f2, x3, f3
        x_mid, f_mid = x1, f1
    else:
        # Should not happen if we started with a bracket and m was in it
        # But if f(m) was 0 we returned. If f(m) had same sign as both, 
        # then either f(a) and f(b) had same sign (contradiction) or m was outside?
        # No, m is inside. So f(a)*f(m)<0 OR f(m)*f(b)<0.
        # So this branch is unreachable unless numerical issues.
        return best_x

    max_hybrid_iter = 100
    for _ in range(max_hybrid_iter):
        if time.time() > deadline:
            break
            
        if abs(f_lo) <= tol_final:
            return x_lo
        if abs(f_hi) <= tol_final:
            return x_hi
        if abs(f_mid) <= tol_final:
            return x_mid
            
        if abs(x_hi - x_lo) < 1e-15:
            return (x_lo + x_hi) / 2.0

        # Try Inverse Quadratic Interpolation
        # Fit quadratic x(f) through (f_lo, x_lo), (f_mid, x_mid), (f_hi, x_hi)
        # and find x where f=0.
        
        # Lagrange basis for f=0
        # x(0) = x_lo * (0-f_mid)(0-f_hi) / ((f_lo-f_mid)(f_lo-f_hi)) + ...
        
        d1 = (f_lo - f_mid) * (f_lo - f_hi)
        d2 = (f_mid - f_lo) * (f_mid - f_hi)
        d3 = (f_hi - f_lo) * (f_hi - f_mid)
        
        x_iqi = None
        if abs(d1) > 1e-30 and abs(d2) > 1e-30 and abs(d3) > 1e-30:
            x_iqi = (x_lo * f_mid * f_hi / d1 +
                     x_mid * f_lo * f_hi / d2 +
                     x_hi * f_lo * f_mid / d3)
        
        # Check if x_iqi is inside the bracket (x_lo, x_hi)
        # Adding a small margin to avoid floating point issues
        margin = 1e-12 * max(1.0, abs(x_hi - x_lo))
        if (x_iqi is not None and math.isfinite(x_iqi) and 
            (x_lo + margin < x_iqi < x_hi - margin)):
            x_new = x_iqi
        else:
            # Bisection
            x_new = (x_lo + x_hi) / 2.0
            
        f_new = f(x_new)
        
        if not math.isfinite(f_new):
            # Fallback to bisection
            x_new = (x_lo + x_hi) / 2.0
            f_new = f(x_new)
            if not math.isfinite(f_new):
                break
                
        if abs(f_new) <= tol_final:
            return x_new
            
        # Update bracket
        if f_lo * f_new < 0:
            # Root in [x_lo, x_new]
            x_hi, f_hi = x_new, f_new
        else:
            # Root in [x_new, x_hi]
            x_lo, f_lo = x_new, f_new
            
        # Update x_mid, f_mid
        # The new point is x_new.
        # The old midpoint x_mid is now the "outer" point.
        # The new midpoint should be the point NOT in the bracket.
        # So if we updated x_hi, the new x_mid is the old x_mid.
        # If we updated x_lo, the new x_mid is the old x_mid.
        # Wait, x_mid is the point outside the current bracket [x_lo, x_hi].
        # So x_mid stays the same value, but its role might change?
        # No, x_mid is just the third point.
        # We need to ensure we have 3 distinct points for IQI.
        # x_mid remains the point that is not x_lo or x_hi.
        # So x_mid, f_mid do not change value, they just represent the "other" point.
        
        # However, if x_new is very close to x_mid, IQI might be unstable.
        # The bracket shrinking handles this.
        
        # One optimization: if the bracket is very small, just return the midpoint
        if abs(x_hi - x_lo) < 1e-10:
             # Convergence is effectively complete or stuck
             # Return the one with smallest |f|
             candidates = [(x_lo, f_lo), (x_hi, f_hi), (x_mid, f_mid)]
             best_local = min(candidates, key=lambda p: abs(p[1]))
             if abs(best_local[1]) <= tol_final:
                 return best_local[0]
             # If not, continue? Or break?
             # If interval is tiny, bisection won't help much.
             break

    # Return best found
    candidates = [(x_lo, f_lo), (x_hi, f_hi), (x_mid, f_mid), (best_x, best_f)]
    best_final = min(candidates, key=lambda p: abs(p[1]))
    return best_final[0]

# EVOLVE-BLOCK-END
