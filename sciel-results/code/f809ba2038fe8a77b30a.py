# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses Brent's method (inverse quadratic interpolation + bisection) for robustness
    and efficiency.
    """
    DEADLINE = 115.0
    START_TIME = time.time()
    
    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE

    def is_finite(x):
        return math.isfinite(x)

    def safe_f(x):
        try:
            fx = f(x)
            if is_finite(fx):
                return fx
        except:
            pass
        return None

    x = float(x0)
    fx = safe_f(x)
    
    if fx is None:
        return x

    if abs(fx) <= 1e-8:
        return x

    # Try to bracket the root
    # We need to find a, b such that f(a) * f(b) < 0
    # Start by expanding interval from x0
    a, b = x, x
    fa, fb = fx, fx
    
    # Initial step size
    delta = 1e-3 * max(1.0, abs(x))
    
    # Try expanding in both directions to find a sign change
    found_bracket = False
    max_expansions = 100
    
    for i in range(max_expansions):
        if not check_deadline():
            break
            
        # Try expanding to the right
        b = x + delta
        fb = safe_f(b)
        if fb is not None:
            if abs(fb) <= 1e-8:
                return b
            if fa * fb < 0:
                found_bracket = True
                break
            # If we didn't find a sign change, keep expanding right if f(x) and f(b) have same sign
            # But if f(b) is much larger in magnitude, maybe we went too far? 
            # For polynomials, usually expanding is fine.
        
        # Try expanding to the left
        a = x - delta
        fa_new = safe_f(a)
        if fa_new is not None:
            if abs(fa_new) <= 1e-8:
                return a
            # Check bracket with original x (which is effectively 'b' in this context if we reset)
            # Actually, let's maintain a and b properly.
            # Let's reset a and b to the current best candidates.
            # If fa_new * fx < 0, we have a bracket [a, x]
            if fa_new * fx < 0:
                a, b = a, x
                fa, fb = fa_new, fx
                found_bracket = True
                break
            # If fa_new * fb < 0 (if fb was valid), we have bracket [a, b]
            if fb is not None and fa_new * fb < 0:
                a, b = a, b
                fa, fb = fa_new, fb
                found_bracket = True
                break
            
            # Update fa to the new left point for next iteration if we continue?
            # No, let's just expand the window.
            # If no sign change, we should probably expand the window further.
            # But we need to be careful not to waste too many calls.
            # Let's just double the delta and try again.
            # However, we need to keep track of the best point so far.
            # For now, if no bracket found, just increase delta.
        
        delta *= 2.0
        
    if not found_bracket:
        # If we couldn't bracket, fall back to a local minimization or secant-like approach?
        # Or just return the best point found.
        # Let's try one more thing: if we have two points with same sign, maybe we can't bracket easily.
        # But for smooth polynomials, if x0 is near a root, expanding should work.
        # If it fails, return the point with smallest |f(x)|.
        # We need to track best_x and best_fx during expansion.
        # Let's re-implement expansion to track best.
        
        # Reset and try again with tracking
        a, b = x, x
        fa, fb = fx, fx
        best_x = x
        best_fx = fx
        delta = 1e-3 * max(1.0, abs(x))
        
        for i in range(max_expansions):
            if not check_deadline():
                break
                
            b = x + delta
            fb = safe_f(b)
            if fb is not None:
                if abs(fb) < abs(best_fx):
                    best_x, best_fx = b, fb
                if fa * fb < 0:
                    a, b = x, b # Wait, a should be x? No, a is left, b is right.
                    # In this loop, we are expanding from x.
                    # If fa * fb < 0, then bracket is [x, b] if fa is fx.
                    # But we might have updated fa? No, fa is still fx in this simplified loop.
                    # Let's just say bracket is [x, b].
                    a, b = x, b
                    fa, fb = fx, fb
                    found_bracket = True
                    break
            
            a = x - delta
            fa_new = safe_f(a)
            if fa_new is not None:
                if abs(fa_new) < abs(best_fx):
                    best_x, best_fx = a, fa_new
                if fa_new * fx < 0:
                    a, b = a, x
                    fa, fb = fa_new, fx
                    found_bracket = True
                    break
                if fb is not None and fa_new * fb < 0:
                    a, b = a, b
                    fa, fb = fa_new, fb
                    found_bracket = True
                    break
                    
            delta *= 2.0
            
        if not found_bracket:
            return best_x

    # Now we have a bracket [a, b] with fa * fb < 0
    # Implement Brent's Method
    # Standard Brent's method variables:
    # a, b: current bracket
    # c: previous b
    # fa, fb, fc: f(a), f(b), f(c)
    # d, e: previous step, step size
    
    a, b = a, b
    fa, fb = fa, fb
    c = b
    fc = fb
    d = b - a
    e = d
    
    tol = 1e-8
    rtol = 1e-12
    
    for _ in range(100):
        if not check_deadline():
            break
            
        if abs(fb) <= tol:
            return b
        if fa == fb:
            # Converged or stuck
            return b if abs(fb) < abs(fa) else a
            
        # Check for convergence
        if abs(b - a) <= rtol * max(1.0, abs(b)):
            return b
            
        # Inverse quadratic interpolation
        if a != c and b != c:
            # Lagrange interpolation
            # S = fb * fa * (b-c) / ((fa-fb)*(fa-fc)) + ...
            # Let's use the standard formula
            s = fb * fa * (b - c) / ((fa - fb) * (fa - fc))
            if b != a:
                s += fb * fb * (a - c) / ((fb - fa) * (fb - fc))
                s += fb * fc * (a - b) / ((fc - fa) * (fc - fb))
            # s is the next point x = b + s? Or is s the numerator for the step?
            # Actually, the formula for x_new is:
            # x = b + ( ... )
            # Let's use a simpler check. If the interpolation point is within the bracket, use it.
            # Otherwise, use bisection.
            
            # Alternative simpler implementation of Brent's step:
            # If the interpolation point x_i is in (a,b) and |x_i - b| < 0.5 * |b - c|, use it.
            # Else use bisection.
            
            # Let's calculate x_i
            try:
                denom1 = (fa - fb) * (fa - fc)
                denom2 = (fb - fa) * (fb - fc)
                denom3 = (fc - fa) * (fc - fb)
                
                if denom1 == 0 or denom2 == 0 or denom3 == 0:
                    raise ZeroDivisionError
                
                term1 = fb * fa * (b - c) / denom1
                term2 = fb * fb * (a - c) / denom2
                term3 = fb * fc * (a - b) / denom3
                
                x_i = b + term1 + term2 + term3
            except:
                x_i = None
            
            if x_i is not None and is_finite(x_i):
                # Check if x_i is in bracket and close enough to b
                if (a < x_i < b or b < x_i < a) and abs(x_i - b) < 0.5 * abs(b - c):
                    s = x_i - b
                else:
                    s = (a + b) / 2.0 - b # Bisection
            else:
                s = (a + b) / 2.0 - b # Bisection
        else:
            # Bisection
            s = (a + b) / 2.0 - b
            
        # Update
        d = e
        e = s
        b = b + s
        fb = safe_f(b)
        
        if fb is None:
            # If evaluation fails, fall back to bisection? Or break?
            # If we just broke out of a good bracket, returning the best known is risky.
            # Let's just break and return the best of a, b, c.
            break
            
        if abs(fb) <= tol:
            return b
            
        if fa * fb < 0:
            c = a
            fc = fa
            a = b
            fa = fb
        else:
            c = b
            fc = fb
            b = a # Wait, standard Brent swaps a and b if needed?
            # Actually, in Brent's method:
            # If f(a) and f(b) have the same sign, we swap a and c? No.
            # We maintain a bracket [a, b] where f(a)*f(b) < 0.
            # If the new point b_new has the same sign as f(b), then the root is in [a, b_old].
            # So we set c = b, b = a? No.
            # Let's trace:
            # We have [a, b]. We pick s. New point is b_new = b + s.
            # If f(a) * f(b_new) < 0, root is in [a, b_new]. Set c=a, a=b_new? No, a stays a, b becomes b_new.
            #   Wait, if f(a)*f(b_new) < 0, then b_new is the new b. So c=a, fc=fa, a=a (unchanged?), b=b_new, fb=f(b_new).
            #   Actually, standard algorithm:
            #   If f(a) * f(b) < 0:
            #       c = a; fc = fa
            #       a = b; fa = fb  <-- This is wrong. a should remain the endpoint with the opposite sign to the NEW b.
            #   Let's use a clear implementation.
            
            # Clear Implementation:
            # We want to maintain a bracket [a, b] such that f(a)*f(b) <= 0.
            # We have a candidate point x_new.
            # If f(a) * f(x_new) < 0:
            #    The root is in [a, x_new].
            #    Set c = b, fc = fb (save old b)
            #    Set b = x_new, fb = f(x_new)
            # Else:
            #    The root is in [x_new, b].
            #    Set c = a, fc = fa (save old a)
            #    Set a = x_new, fa = f(x_new)
            
            # Let's re-evaluate the logic in the loop above.
            # I used:
            # if fa * fb < 0: # fa is f(a), fb is f(b_new)
            #    c = a; fc = fa
            #    a = b; fa = fb  <-- This is WRONG. If fa * fb < 0, then a and b_new have opposite signs.
            #    The bracket is [a, b_new]. So a stays a, b becomes b_new.
            #    So: c = b (old b), fc = fb (old fb), b = b_new, fb = f(b_new).
            #    Wait, my variable names in the code above were confusing.
            #    Let's rewrite the update step clearly.

        # Let's restart the Brent loop with clear variable names to avoid confusion.
        # I will break out and restart the Brent part correctly.
        break

    # Correct Brent's Method Implementation
    # We have a bracket [a, b] with fa * fb < 0
    # We need to ensure we track the best point.
    
    # Re-initialize for correct Brent
    # We might have messed up a, b in the previous failed attempt.
    # Let's just re-bracket if necessary or assume the first bracketing phase succeeded.
    # To be safe, let's re-run the bracketing logic if we broke out early due to confusion.
    # But for efficiency, let's assume the first `found_bracket` was correct and a, b, fa, fb are valid.
    
    # However, the update logic in the first Brent loop was flawed.
    # Let's implement a clean Brent's method from scratch in this function,
    # using the bracket found.
    
    # If we didn't find a bracket in the first pass, we returned best_x.
    # So here we assume found_bracket is True and a, b, fa, fb are set.
    
    # Clean Brent's
    c = b
    fc = fb
    d = b - a
    e = d
    
    for _ in range(100):
        if not check_deadline():
            break
            
        if abs(fb) <= tol:
            return b
        if fa == fb:
            return b if abs(fb) < abs(fa) else a
            
        if abs(b - a) <= rtol * max(1.0, abs(b)):
            return b
            
        # Inverse Quadratic Interpolation
        if a != c and b != c:
            try:
                s = fb * fa * (b - c) / ((fa - fb) * (fa - fc))
                s += fb * fb * (a - c) / ((fb - fa) * (fb - fc))
                s += fb * fc * (a - b) / ((fc - fa) * (fc - fb))
                x_new = b + s
            except:
                x_new = None
        else:
            x_new = None
            
        # Bisection fallback
        if x_new is None or not is_finite(x_new) or not ((a < x_new < b) or (b < x_new < a)) or abs(x_new - b) >= 0.5 * abs(b - c):
            x_new = (a + b) / 2.0
            
        # Evaluate
        f_new = safe_f(x_new)
        if f_new is None:
            break
            
        if abs(f_new) <= tol:
            return x_new
            
        # Update bracket
        if fa * f_new < 0:
            c = b
            fc = fb
            b = x_new
            fb = f_new
        else:
            c = a
            fc = fa
            a = x_new
            fa = f_new
            
        d = b - a
        e = x_new - b # step size
        
    # Return best of a, b, c
    candidates = [(a, fa), (b, fb), (c, fc)]
    best_x, best_fx = min(candidates, key=lambda p: abs(p[1]))
    return best_x

# EVOLVE-BLOCK-END
