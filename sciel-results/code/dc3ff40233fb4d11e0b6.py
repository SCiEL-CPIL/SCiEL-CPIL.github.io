import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a hybrid approach:
    1. Initial bracketing search if necessary.
    2. Secant method for rapid convergence.
    3. Inverse Quadratic Interpolation (IQI) for superlinear convergence when conditions are met.
    4. Bisection fallback for robustness.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    x0 = float(x0)
    max_evals = 500
    evals_used = 0
    tol = 1e-8
    
    # State variables
    best_x = x0
    best_f = None
    
    def update_best(x, fx):
        nonlocal best_x, best_f
        if best_f is None or abs(fx) < abs(best_f):
            best_x = x
            best_f = fx

    # Initial evaluation
    f0 = f(x0)
    evals_used += 1
    update_best(x0, f0)
    
    if abs(f0) <= tol:
        return x0

    # We need a second point to start secant/interpolation.
    # Try a small perturbation to get x1.
    h = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + h
    f1 = f(x1)
    evals_used += 1
    update_best(x1, f1)
    
    if abs(f1) <= tol:
        return x1

    # Maintain a bracket [a, b] if a sign change exists
    # Initialize bracket with the two points if they have opposite signs
    if f0 * f1 < 0:
        a, b = x0, x1
        fa, fb = f0, f1
    else:
        # No sign change yet. Try to expand the bracket or just proceed with secant.
        # For ill-conditioned problems, a sign change might not exist nearby, 
        # or the function might be non-monotonic.
        # We will track the last two points for secant, and keep a bracket if one appears.
        a, b = None, None
        fa, fb = None, None
        # Keep the two points with the smallest absolute function values as our "active" secant pair?
        # Or just use the most recent two. Secant usually works best with recent history.
        # Let's stick to the most recent two points for the secant step, 
        # but keep a global bracket if found.
        
    # History for interpolation: (x, f) pairs
    # We keep the last 3 points for IQI
    history = [(x0, f0), (x1, f1)]
    
    # Current active points for Secant
    x_prev, f_prev = x0, f0
    x_curr, f_curr = x1, f1
    
    for i in range(max_evals - 2):
        if time.time() > deadline:
            break
            
        # Check convergence
        if abs(f_curr) <= tol:
            return x_curr
        if abs(f_prev) <= tol:
            return x_prev
            
        # Update best
        update_best(x_curr, f_curr)
        update_best(x_prev, f_prev)
        
        # Determine next step
        x_next = None
        f_next = None
        
        # Strategy:
        # 1. If we have a bracket [a, b] and the secant/IQI step falls outside or is too large, use Bisection.
        # 2. If we have 3 points, try Inverse Quadratic Interpolation.
        # 3. Otherwise, use Secant.
        
        use_bisection = False
        
        if a is not None and b is not None:
            # Check if current points are within bracket or if we should bisect
            # Standard Brent's method logic:
            # If the secant step is too large or goes outside the bracket, bisect.
            pass # Logic handled below
        
        # Calculate Secant step
        if f_curr != f_prev:
            x_secant = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        else:
            x_secant = None
            
        # Calculate IQI step if we have 3 points
        x_iqi = None
        if len(history) >= 3:
            x0_h, f0_h = history[-3]
            x1_h, f1_h = history[-2]
            x2_h, f2_h = history[-1]
            
            # Inverse Quadratic Interpolation
            # Solves for x given f(x) = 0 using Lagrange interpolation on (f, x)
            # Formula:
            # x = (f^2 * (x1/(f*(f-f2)) + x2/(f2*(f-f1))) ) ... wait, standard formula is:
            # x = (f1^2 * x2 / ((f1-f0)(f1-f2)) + f2^2 * x1 / ((f2-f0)(f2-f1))) / 
            #     (f1 * f2 / ((f1-f0)(f1-f2)) + f2 * f1 / ((f2-f0)(f2-f1))) -- This is messy.
            
            # Simpler IQI formula:
            # Let (x0, f0), (x1, f1), (x2, f2) be the points.
            # We want x such that P(f(x)) = 0 where P is quadratic in f.
            # x = [ f^2 * ( x1 / (f*(f-f2)) + x2 / (f2*(f-f1)) ) ] ... No.
            
            # Correct IQI Formula:
            # x = ( f1^2 * x2 / ((f1-f0)*(f1-f2)) + f2^2 * x1 / ((f2-f0)*(f2-f1)) ) /
            #     ( f1 * f2 / ((f1-f0)*(f1-f2)) + f2 * f1 / ((f2-f0)*(f2-f1)) )
            # Wait, the denominator is sum of weights for x0? No, x0 is not in numerator?
            # The formula is:
            # x = [ f1^2 * x2 / ((f1-f0)(f1-f2)) + f2^2 * x1 / ((f2-f0)(f2-f1)) ] /
            #     [ f1 * f2 / ((f1-f0)(f1-f2)) + f2 * f1 / ((f2-f0)(f2-f1)) ]
            # Actually, the standard form is:
            # x = ( f1^2 * x2 / (f1*(f1-f2)) + f2^2 * x1 / (f2*(f2-f1)) ) / ( f1 / (f1-f2) + f2 / (f2-f1) )
            # Let's use the stable form from Numerical Recipes or similar.
            
            # x = (f1^2 * x2 / ((f1-f0)*(f1-f2)) + f2^2 * x1 / ((f2-f0)*(f2-f1))) / 
            #     (f1*f2 / ((f1-f0)*(f1-f2)) + f2*f1 / ((f2-f0)*(f2-f1)))
            # Note: x0 is missing from numerator? No, the term for x0 is f0^2 * x0 / ... but we are solving for x where f=0.
            # The interpolation is x = c0 + c1*f + c2*f^2.
            # We want x(0).
            # x(0) = c0.
            # Lagrange basis:
            # L0(f) = f*(f-f2) / ((f0-f1)*(f0-f2))
            # L1(f) = f*(f-f0) / ((f1-f0)*(f1-f2))
            # L2(f) = f*(f-f1) / ((f2-f0)*(f2-f1))
            # x(0) = x0*L0(0) + x1*L1(0) + x2*L2(0)
            # L0(0) = 0 * (-f2) / ... = 0. So x0 term vanishes?
            # Yes, for inverse quadratic interpolation, the point with the largest residual (usually f0 if it's old) drops out of the numerator for the root estimate if we assume f(0)=0?
            # Actually, L0(0) = 0. L1(0) = 0*(-f0) / ((f1-f0)*(f1-f2)) = 0. L2(0) = 0.
            # This implies x(0) = 0? That's wrong.
            # The interpolation is x = P(f). We want P(0).
            # P(f) = x0 * L0(f) + x1 * L1(f) + x2 * L2(f)
            # L0(f) = (f - f1)(f - f2) / ((f0 - f1)(f0 - f2))
            # L1(f) = (f - f0)(f - f2) / ((f1 - f0)(f1 - f2))
            # L2(f) = (f - f0)(f - f1) / ((f2 - f0)(f2 - f1))
            # At f=0:
            # L0(0) = f1*f2 / ((f0-f1)(f0-f2))
            # L1(0) = f0*f2 / ((f1-f0)(f1-f2))
            # L2(0) = f0*f1 / ((f2-f0)(f2-f1))
            
            denom1 = (f0_h - f1_h) * (f0_h - f2_h)
            denom2 = (f1_h - f0_h) * (f1_h - f2_h)
            denom3 = (f2_h - f0_h) * (f2_h - f1_h)
            
            if abs(denom1) > 1e-300 and abs(denom2) > 1e-300 and abs(denom3) > 1e-300:
                L0_0 = (f1_h * f2_h) / denom1
                L1_0 = (f0_h * f2_h) / denom2
                L2_0 = (f0_h * f1_h) / denom3
                x_iqi = x0_h * L0_0 + x1_h * L1_0 + x2_h * L2_0
            
        # Decide which step to take
        # If we have a bracket, check if IQI or Secant step is inside.
        # If not, use Bisection.
        
        if a is not None and b is not None:
            # Ensure a < b
            if a > b:
                a, b = b, a
                fa, fb = fb, fa
            
            # Check if x_iqi is within bracket
            valid_iqi = False
            if x_iqi is not None:
                if a < x_iqi < b:
                    # Also check if it's "close" enough to not be too aggressive?
                    # For now, if it's inside, it's a candidate.
                    valid_iqi = True
            
            valid_secant = False
            if x_secant is not None:
                if a < x_secant < b:
                    valid_secant = True
            
            # Prefer IQI if valid and distinct from secant?
            # Usually IQI is faster.
            if valid_iqi:
                x_next = x_iqi
            elif valid_secant:
                x_next = x_secant
            else:
                # Bisection
                x_next = (a + b) / 2.0
                use_bisection = True
        else:
            # No bracket. Use Secant or IQI.
            # If IQI is available and finite, use it? 
            # IQI can be unstable if points are not well-distributed.
            # Secant is safer without a bracket.
            # Let's use Secant primarily, unless IQI is very close to Secant?
            # Actually, without a bracket, IQI can jump to infinity.
            # We'll stick to Secant if no bracket, unless we detect convergence acceleration.
            # For simplicity and robustness: Use Secant.
            if x_secant is not None:
                x_next = x_secant
            else:
                # Fallback: small step
                x_next = x_curr + 1e-4 * max(1.0, abs(x_curr))
        
        # Evaluate f at x_next
        if x_next is None or math.isnan(x_next) or math.isinf(x_next):
            # If step is invalid, try bisection if bracket exists, else perturb
            if a is not None and b is not None:
                x_next = (a + b) / 2.0
            else:
                x_next = x_curr + 1e-4 * max(1.0, abs(x_curr))
                
        f_next = f(x_next)
        evals_used += 1
        update_best(x_next, f_next)
        
        if abs(f_next) <= tol:
            return x_next
            
        # Update Bracket
        if a is not None and b is not None:
            if fa * f_next <= 0:
                b = x_next
                fb = f_next
            else:
                a = x_next
                fa = f_next
        else:
            # Check if we can establish a bracket with the last two points
            if f_prev * f_next < 0:
                a, b = x_prev, x_next
                fa, fb = f_prev, f_next
            elif f_curr * f_next < 0:
                a, b = x_curr, x_next
                fa, fb = f_curr, f_next
            
        # Update History
        history.append((x_next, f_next))
        if len(history) > 3:
            history.pop(0)
            
        # Update Secant Points
        # We want the two points with the smallest |f| to be our secant pair?
        # Or just the most recent?
        # Using the two most recent is standard Secant.
        # However, if one has a much larger |f|, it might be better to drop it.
        # Let's stick to most recent for simplicity, as IQI uses history anyway.
        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_next, f_next
        
    return best_x
# EVOLVE-BLOCK-END
