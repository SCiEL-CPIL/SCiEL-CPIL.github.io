"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Robust scalar root finder using Illinois Regula Falsi.
    This method is distinct from the previous Bisection/Secant hybrid.
    It guarantees convergence if a bracket is found and is typically faster
    than pure bisection.
    """
    x = float(x0)
    tol = 1e-8
    max_calls = 450
    
    best_x = x
    best_fx = f(x)
    call_count = 1
    
    if abs(best_fx) <= tol:
        return best_x
    
    # Find a bracket
    a = x
    fa = best_fx
    found_bracket = False
    step = 1.0
    
    for i in range(20):
        if call_count >= max_calls - 5:
            break
        
        # Try positive
        b = a + step
        fb = f(b)
        call_count += 1
        if abs(fb) < abs(best_fx):
            best_x = b
            best_fx = fb
        if abs(fb) <= tol:
            return b
        if fa * fb < 0:
            found_bracket = True
            break
        
        # Try negative
        b2 = a - step
        fb2 = f(b2)
        call_count += 1
        if abs(fb2) < abs(best_fx):
            best_x = b2
            best_fx = fb2
        if abs(fb2) <= tol:
            return b2
        if fa * fb2 < 0:
            a = b2
            fa = fb2
            b = x
            fb = best_fx
            found_bracket = True
            break
            
        step *= 2.0

    if found_bracket:
        # Illinois Regula Falsi
        # Ensure |fa| > |fb|
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
            
        for i in range(400):
            if call_count >= max_calls:
                break
            
            denom = fb - fa
            if denom == 0:
                break
                
            x_new = (a * fb - b * fa) / denom
            f_new = f(x_new)
            call_count += 1
            
            if abs(f_new) < abs(best_fx):
                best_x = x_new
                best_fx = f_new
            if abs(f_new) <= tol:
                return x_new
            
            # Update bracket
            if fa * f_new < 0:
                b = x_new
                fb = f_new
            else:
                a = x_new
                fa = f_new
                
            # Illinois Step:
            # Maintain |fa| > |fb|
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa
                
            # If |f_new| < |fb| (the smaller end), shrink the smaller end's side
            # by replacing the larger end with the smaller end and halving the 
            # function value at the smaller end.
            if abs(f_new) < abs(fb):
                a = b
                fa = fb
                fb = fb * 0.5
            else:
                # No special shrink needed, just keep the new bracket
                pass
        
        return best_x
    else:
        # Fallback: Secant
        x_curr = x
        f_curr = best_fx
        h = 1e-4 * max(1.0, abs(x_curr))
        x_prev = x_curr - h
        if call_count < max_calls:
            f_prev = f(x_prev)
            call_count += 1
            if abs(f_prev) < abs(best_fx):
                best_x = x_prev
                best_fx = f_prev
            if abs(f_prev) <= tol:
                return x_prev
        
        for i in range(400):
            if call_count >= max_calls:
                break
            if f_curr == f_prev:
                break
            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            
            if not math.isfinite(x_new) or abs(x_new - x_curr) > 10 * max(1.0, abs(x_curr)):
                x_new = x_curr - 0.5 * h
                h *= 0.5
                if abs(h) < 1e-15:
                    break
            
            if call_count >= max_calls:
                break
            f_new = f(x_new)
            call_count += 1
            if abs(f_new) < abs(best_fx):
                best_x = x_new
                best_fx = f_new
            if abs(f_new) <= tol:
                return x_new
            
            x_prev = x_curr
            f_prev = f_curr
            x_curr = x_new
            f_curr = f_new
        
        return best_x

# EVOLVE-BLOCK-END
