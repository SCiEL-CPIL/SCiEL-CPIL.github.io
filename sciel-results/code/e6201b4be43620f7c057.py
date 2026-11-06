"""Derivative-free trust-region root finder for ill-conditioned 1D problems."""
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using derivative-free methods.
    Strategy:
      1. Try to find a bracket around x0 using adaptive expansion.
      2. If bracket found, use a corrected Brent's method.
      3. If no bracket, use adaptive local minimization of |f(x)|.
      4. Fall back to secant with safeguards.
    
    Budget: at most 500 evaluations of f.
    """
    target_tol = 1e-8
    max_evals = 500
    eval_count = [0]

    def eval_f(x):
        eval_count[0] += 1
        try:
            val = f(x)
            if not (isinstance(val, (int, float)) or (hasattr(val, '__float__'))):
                val = float(val)
            if math.isnan(val) or math.isinf(val):
                return float('inf')
            return val
        except Exception:
            return float('inf')

    def abs_f(x):
        return abs(eval_f(x))

    # --- Phase 1: Attempt to find a bracket around x0 ---
    x = float(x0)
    f0 = eval_f(x)
    if abs(f0) <= target_tol:
        return x

    # Try expanding a bracket with adaptive step sizes
    bracket_found = False
    a, b = None, None
    fa, fb = None, None

    # Adaptive step: start small, grow geometrically
    h = max(1e-8, abs(x) * 1e-8) if abs(x) > 0 else 1e-8
    # Also try some fixed steps for robustness
    fixed_steps = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    
    # First try the adaptive small step
    for h_try in [h] + fixed_steps:
        if eval_count[0] >= max_evals - 10:
            break
        x1 = x + h_try
        x2 = x - h_try
        f1 = eval_f(x1)
        f2 = eval_f(x2)
        
        if abs(f1) <= target_tol:
            return x1
        if abs(f2) <= target_tol:
            return x2
        
        # Check sign changes
        if f0 * f1 < 0:
            a, b = x, x1
            fa, fb = f0, f1
            bracket_found = True
            break
        if f0 * f2 < 0:
            a, b = x2, x
            fa, fb = f2, f0
            bracket_found = True
            break
        if f1 * f2 < 0:
            a, b = x1, x2
            fa, fb = f1, f2
            bracket_found = True
            break

    if bracket_found:
        # --- Phase 2: Use corrected Brent's method on the bracket ---
        def brent(a, b, fa, fb, tol=1e-15, max_iter=200):
            """Corrected Brent's root-finding method."""
            if fa * fb > 0:
                return None
            
            # Ensure |fa| < |fb| so b is the closer to root
            if abs(fa) > abs(fb):
                a, b = b, a
                fa, fb = fb, fa
            
            c = a
            fc = fa
            d = b - a
            e = d
            
            for _ in range(max_iter):
                if eval_count[0] >= max_evals - 5:
                    break
                
                # Convergence check
                if abs(b - a) < tol or abs(fb) <= target_tol:
                    return b
                
                # If sign change between a and c, move a to c
                if fa * fc < 0:
                    a = c
                    fa = fc
                    d = b - a
                    e = d
                
                # If |fc| < |fa|, swap to keep a as the farther point
                if abs(fc) < abs(fa):
                    a = b
                    b = c
                    c = a
                    fa = fb
                    fb = fc
                    fc = fa
                    d = b - a
                    e = d
                
                # Choose next step
                tol1 = 2e-16 * max(1.0, abs(b))
                t = 0.5 * (a + b)
                
                # Check if bisection is needed
                if abs(b - t) <= max(tol1, 1e-14) or abs(fa - fb) <= 1e-30:
                    # Use bisection
                    s = 0.5 * (a - b)
                else:
                    # Try inverse quadratic interpolation
                    if a == c:
                        # Linear interpolation
                        s = (a - b) * fa / (fa - fb) if fa != fb else 0.5 * (a - b)
                    elif b == c:
                        # Linear interpolation
                        s = (a - b) * fb / (fa - fb) if fa != fb else 0.5 * (a - b)
                    else:
                        # Inverse quadratic interpolation
                        num = fa * fc * (fb - fa)
                        den = fa * fb * (fc - fa)
                        if abs(den) > 1e-30:
                            s = num / den
                        else:
                            s = 0.5 * (a - b)
                    
                    # Check if s is reasonable, otherwise use bisection
                    if (s > min(a, b) - 2*abs(b-a) or s > max(a, b) + 2*abs(b-a)) or \
                       (s < 2*abs(t - b) and (b - t) * s >= 0):
                        s = 0.5 * (a - b)
                
                new_x = b + s
                new_f = eval_f(new_x)
                if abs(new_f) <= target_tol:
                    return new_x
                
                c = b
                fc = fb
                b = new_x
                fb = new_f
            
            # Return the best point
            if abs(fa) < abs(fb):
                return a
            return b

        result = brent(a, b, fa, fb)
        if result is not None:
            f_result = eval_f(result)
            if abs(f_result) <= target_tol:
                return result
            # If close enough, return it
            if abs(f_result) < abs(fa) and abs(f_result) < abs(fb):
                return result
        # Return the better of a, b
        if abs(fa) < abs(fb):
            return a
        return b

    # --- Phase 3: No bracket found. Use adaptive local minimization of |f(x)| ---
    # Use a golden-section-like adaptive search
    
    # Initial scale
    scale = max(1e-6, abs(x) * 1e-6) if abs(x) > 0 else 1e-6
    
    # Evaluate at x and two neighbors
    x1 = x + scale
    x2 = x - scale
    f_x = abs_f(x)
    f_x1 = abs_f(x1)
    f_x2 = abs_f(x2)
    
    # Find the best starting point
    candidates = [(x, f_x), (x1, f_x1), (x2, f_x2)]
    candidates.sort(key=lambda p: p[1])
    best_x, best_f = candidates[0]
    
    # Adaptive step search: try to find a direction and step that reduces |f|
    step = scale
    for i in range(min(300, max_evals - eval_count[0])):
        if best_f <= target_tol:
            break
        if eval_count[0] >= max_evals - 5:
            break
        
        # Try steps in both directions
        improved = False
        for direction in [1.0, -1.0]:
            if eval_count[0] >= max_evals - 2:
                break
            cand = best_x + direction * step
            cf = abs_f(cand)
            if cf < best_f * 0.99:  # Require meaningful improvement
                best_x = cand
                best_f = cf
                step *= 0.7  # Shrink step on success
                improved = True
                break
        
        if not improved:
            step *= 2.0
            if step > 1e12:
                break
    
    if best_f <= target_tol:
        return best_x
    
    # --- Phase 4: Try secant method from the best points ---
    # Get the two best distinct points
    points = [(x, f0), (x1, abs_f(x1)), (x2, abs_f(x2)), (best_x, best_f)]
    # Remove duplicates by x value
    seen = set()
    unique_points = []
    for px, pf in points:
        key = round(px, 12)
        if key not in seen:
            seen.add(key)
            unique_points.append((px, pf))
    unique_points.sort(key=lambda p: p[1])
    
    if len(unique_points) >= 2:
        x_a, f_a = unique_points[0]
        x_b, f_b = unique_points[1]
        
        for i in range(min(100, max_evals - eval_count[0])):
            if eval_count[0] >= max_evals - 2:
                break
            if abs(f_b - f_a) < 1e-30:
                break
            x_new = x_b - f_b * (x_b - x_a) / (f_b - f_a)
            # Safeguard
            if abs(x_new - x_b) > 100 * abs(x_b - x_a):
                x_new = x_b + 0.5 * (x_b - x_a)
            f_new = eval_f(x_new)
            if abs(f_new) <= target_tol:
                return x_new
            # Keep the two best points
            if abs(f_new) < abs(f_a):
                x_a, f_a = x_b, f_b
                x_b, f_b = x_new, f_new
            elif abs(f_new) < abs(f_b):
                x_b, f_b = x_new, f_new
            else:
                x_a, f_a = x_b, f_b
                x_b, f_b = x_new, f_new
    
    # Return the best point found
    all_points = [(x, f0), (x1, abs_f(x1)), (x2, abs_f(x2)), (best_x, best_f)]
    all_points.sort(key=lambda p: p[1])
    return all_points[0][0]

# EVOLVE-BLOCK-END
