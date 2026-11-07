"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Derivative-free inverse quadratic interpolation (IQI) root finder.
    Fits a quadratic to the last three (x, f(x)) points and inverts it to predict the next x.
    Falls back to bisection or secant if needed for robustness.
    """
    import math

    x = float(x0)
    
    # Initial step size for building the three points
    # We need three distinct points to fit a quadratic
    # Start by evaluating f(x0), then take a small step
    h_init = 1e-4 * max(1.0, abs(x))
    
    # Store points: list of (x, fx)
    points = []
    
    # Evaluate first point
    fx0 = f(x)
    points.append((x, fx0))
    
    # If already a root
    if abs(fx0) <= 1e-12:
        return x
    
    # Generate second point
    x1 = x + h_init
    fx1 = f(x1)
    points.append((x1, fx1))
    
    if abs(fx1) <= 1e-12:
        return x1
    
    # Generate third point
    x2 = x - h_init
    fx2 = f(x2)
    points.append((x2, fx2))
    
    if abs(fx2) <= 1e-12:
        return x2
    
    # Now we have 3 points. Use IQI.
    # To ensure we have a bracket if possible, check signs.
    # But IQI doesn't strictly require a bracket, just distinct x's.
    # However, for polynomials, if we have a bracket, we can use it to ensure convergence.
    # Let's maintain a bracket if one exists, otherwise just use the 3 most recent points.
    
    # Sort points by x for stability? Not strictly necessary for IQI formula if we use the general form.
    # Let's use the last 3 points in the list.
    
    max_iter = 60
    eps = 1e-12
    
    for _ in range(max_iter):
        # Check if any of the last 3 points is a root
        for px, pfx in points[-3:]:
            if abs(pfx) <= eps:
                return px
        
        # Take the last 3 points
        x_a, f_a = points[-3]
        x_b, f_b = points[-2]
        x_c, f_c = points[-1]
        
        # Ensure x values are distinct
        if abs(x_a - x_b) < 1e-15 or abs(x_b - x_c) < 1e-15 or abs(x_a - x_c) < 1e-15:
            # Points too close, perturb
            h = 1e-4 * max(1.0, abs(x_c))
            x_new = x_c + h
            f_new = f(x_new)
            points.append((x_new, f_new))
            if len(points) > 5:
                points.pop(0)
            continue
        
        # Inverse Quadratic Interpolation
        # We fit a quadratic polynomial p(t) = A*t^2 + B*t + C through (x_a, f_a), (x_b, f_b), (x_c, f_c)
        # Then solve p(t) = 0.
        # Alternatively, fit a quadratic q(f) = alpha*f^2 + beta*f + gamma through (f_a, x_a), (f_b, x_b), (f_c, x_c)
        # and solve q(f) = 0 for f, then x = q(0)? No, that's not right.
        # IQI: Fit a quadratic x = q(f) = A*f^2 + B*f + C through (f_a, x_a), (f_b, x_b), (f_c, x_c).
        # Then the next x is q(0) = C.
        
        # Let's compute A, B, C.
        # A*f_a^2 + B*f_a + C = x_a
        # A*f_b^2 + B*f_b + C = x_b
        # A*f_c^2 + B*f_c + C = x_c
        
        # Subtract to eliminate C:
        # A*(f_a^2 - f_c^2) + B*(f_a - f_c) = x_a - x_c
        # A*(f_b^2 - f_c^2) + B*(f_b - f_c) = x_b - x_c
        
        # Let d_a = f_a - f_c, d_b = f_b - f_c
        # A*(d_a*(f_a+f_c)) + B*d_a = x_a - x_c
        # A*(d_b*(f_b+f_c)) + B*d_b = x_b - x_c
        
        d_a = f_a - f_c
        d_b = f_b - f_c
        
        if abs(d_a) < 1e-15 or abs(d_b) < 1e-15:
            # Degenerate case, use secant method on the two points with largest |f|
            if abs(f_a) > abs(f_b):
                # Use b and c
                if abs(f_c) < 1e-15:
                    return x_c
                x_new = x_c - f_c * (x_b - x_c) / (f_b - f_c)
            else:
                # Use a and c
                if abs(f_c) < 1e-15:
                    return x_c
                x_new = x_c - f_c * (x_a - x_c) / (f_a - f_c)
        else:
            # Solve linear system for A and B
            # Let M1 = d_a*(f_a+f_c), M2 = d_b*(f_b+f_c)
            # M1*A + d_a*B = x_a - x_c
            # M2*A + d_b*B = x_b - x_c
            
            M1 = d_a * (f_a + f_c)
            M2 = d_b * (f_b + f_c)
            R1 = x_a - x_c
            R2 = x_b - x_c
            
            det = M1 * d_b - M2 * d_a
            
            if abs(det) < 1e-15:
                # Degenerate, use secant
                if abs(f_a) > abs(f_b):
                    x_new = x_c - f_c * (x_b - x_c) / (f_b - f_c)
                else:
                    x_new = x_c - f_c * (x_a - x_c) / (f_a - f_c)
            else:
                A = (R1 * d_b - R2 * d_a) / det
                B = (M1 * R2 - M2 * R1) / det
                C = x_c - A * f_c * f_c - B * f_c
                x_new = C
        
        # Check if x_new is finite
        if not math.isfinite(x_new):
            # Fallback to secant
            if abs(f_b) > abs(f_c):
                x_new = x_c - f_c * (x_a - x_c) / (f_a - f_c)
            else:
                x_new = x_c - f_c * (x_b - x_c) / (f_b - f_c)
        
        # Evaluate f(x_new)
        f_new = f(x_new)
        points.append((x_new, f_new))
        
        # Keep only the last 3 points? Or the 3 best?
        # For IQI, using the 3 most recent is standard.
        # But if we have a bracket, we should keep the bracket points.
        # Let's check if we have a bracket among the last 3 points.
        # If f_a * f_c <= 0, then [x_a, x_c] is a bracket.
        # We should keep the points that form the bracket and the new point.
        
        # Simple strategy: keep the last 3 points.
        if len(points) > 3:
            points.pop(0)
            
        if abs(f_new) <= 1e-8:
            return x_new
            
        # Safety: if x_new is too close to existing points, perturb
        if min(abs(x_new - p[0]) for p in points[:-1]) < 1e-10 * max(1.0, abs(x_new)):
            h = 1e-4 * max(1.0, abs(x_new))
            x_pert = x_new + h
            f_pert = f(x_pert)
            points.append((x_pert, f_pert))
            if len(points) > 3:
                points.pop(0)
            if abs(f_pert) <= 1e-8:
                return x_pert

    # Final check
    for px, pfx in points:
        if abs(pfx) <= 1e-8:
            return px
    
    # Return the point with smallest |f(x)|
    return min(points, key=lambda p: abs(p[1]))[0]
# EVOLVE-BLOCK-END
