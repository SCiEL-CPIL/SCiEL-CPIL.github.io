import math

def find_root(f, x0):
    tol = 1e-8
    x = float(x0)
    fx = f(x)
    
    if abs(fx) <= tol:
        return x

    # Phase 1: Find a bracket [a, b] such that f(a) * f(b) < 0
    # We expand from x0 in both directions.
    a = x
    fa = fx
    b = None
    fb = None
    
    # Try small steps first to catch roots close to x0
    # Then larger steps
    steps = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]
    
    # We keep the best point (smallest |f|) found during search
    x_best = x
    fx_best = fx

    for delta in steps:
        # Try x + delta
        x_try = x + delta
        f_try = f(x_try)
        if not math.isfinite(f_try):
            continue
        if abs(f_try) < abs(fx_best):
            x_best = x_try
            fx_best = f_try
        if abs(f_try) <= tol:
            return x_try
        if fa * f_try < 0:
            b = x_try
            fb = f_try
            break
        
        # Try x - delta
        x_try = x - delta
        f_try = f(x_try)
        if not math.isfinite(f_try):
            continue
        if abs(f_try) < abs(fx_best):
            x_best = x_try
            fx_best = f_try
        if abs(f_try) <= tol:
            return x_try
        if fa * f_try < 0:
            b = x_try
            fb = f_try
            break

    if b is None:
        # No bracket found. Return best point found.
        # This is a fallback for non-crossing roots or if range is too small.
        return x_best

    # Phase 2: Illinois Algorithm (Robust False Position)
    # Ensure a is the one with smaller |f|? No, Illinois swaps the one with LARGER |f|.
    x_a = a
    f_a = fa
    x_b = b
    f_b = fb

    # Ensure we start with a and b correctly
    for _ in range(100):
        if abs(f_a) <= tol:
            return x_a
        if abs(f_b) <= tol:
            return x_b

        # Illinois update:
        # If |f_a| > |f_b|, halve the distance of a to b?
        # Standard Illinois:
        # Compute root estimate x_new = x_b - f_b * (x_b - x_a) / (f_b - f_a)
        # If |f_a| > |f_b|, set f_a = f_a / 2 (effectively moving a towards b)
        # Else set f_b = f_b / 2
        
        if f_a == f_b:
            # Degenerate case, bisection
            x_new = (x_a + x_b) / 2.0
        else:
            x_new = x_b - f_b * (x_b - x_a) / (f_b - f_a)
            
        f_new = f(x_new)
        
        if abs(f_new) <= tol:
            return x_new

        if f_a * f_new < 0:
            # Root is in [x_a, x_new]
            x_b = x_new
            f_b = f_new
            # Illinois rule: if |f_a| > |f_b| (new), halve f_a
            if abs(f_a) > abs(f_b):
                f_a = f_a / 2.0
        else:
            # Root is in [x_new, x_b]
            x_a = x_new
            f_a = f_new
            # Illinois rule: if |f_b| > |f_a| (new), halve f_b
            if abs(f_b) > abs(f_a):
                f_b = f_b / 2.0

        # Check convergence of interval
        if abs(x_b - x_a) < 1e-12:
            return x_a if abs(f_a) < abs(f_b) else x_b

    return x_a if abs(f_a) < abs(f_b) else x_b
