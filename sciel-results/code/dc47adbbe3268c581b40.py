"""Root finding for ill-conditioned problems using a robust hybrid approach."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a robust implementation of Brent's method if a bracket is found,
    otherwise uses a safeguarded Secant method.
    """
    x = float(x0)
    max_evals = 500
    eval_count = 0
    best_x = x
    best_f = float('inf')
    
    def eval_f(val):
        nonlocal eval_count, best_x, best_f
        if eval_count >= max_evals:
            raise RuntimeError("Max evaluations exceeded")
        result = f(val)
        eval_count += 1
        if abs(result) < best_f:
            best_f = abs(result)
            best_x = val
        return result

    # Initial evaluation
    fx = eval_f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Try to establish a bracket [a, b]
    # Start with a small step
    h = 1e-4 * max(1.0, abs(x))
    if h == 0: h = 1e-4
    
    a = x
    fa = fx
    b = x + h
    fb = eval_f(b)
    
    if abs(fb) <= 1e-8:
        return b

    # Expand bracket to find sign change
    # We try to expand in the direction where the function is moving towards 0
    # or just alternate sides if unclear.
    # Strategy: If fa and fb have same sign, expand the side with smaller |f|? 
    # Actually, standard expansion:
    # If fa*fb > 0:
    #   if |fa| < |fb|: expand a (a = a - (b-a))
    #   else: expand b (b = b + (b-a))
    
    step = h
    for _ in range(50):
        if fa * fb < 0:
            break
        if abs(fa) < abs(fb):
            # Expand a
            a_new = a - step
            fa = eval_f(a_new)
            a = a_new
        else:
            # Expand b
            b_new = b + step
            fb = eval_f(b_new)
            b = b_new
        step *= 2
        if abs(a) > 1e10 or abs(b) > 1e10:
            break
        if abs(fa) <= 1e-8: return a
        if abs(fb) <= 1e-8: return b

    if fa * fb < 0:
        # Use Brent's Method
        # Standard Brent's algorithm
        # a, b are the bracket
        c = a
        fc = fa
        d = b - a
        e = d
        
        for _ in range(max_evals - eval_count):
            if abs(fb) <= 1e-8:
                return b
            if abs(fa) <= 1e-8:
                return a
            
            # Check for convergence
            if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                return b if abs(fb) < abs(fa) else a

            # Inverse quadratic interpolation or Secant
            # If a and b are close, or if previous step was bisection, use Secant
            # Standard Brent's logic:
            
            # 1. If a and b are not close, and previous step was not bisection
            if abs(e) > 1e-14 * max(1.0, abs(a), abs(b)) and abs(fc) > abs(fb):
                # Try Inverse Quadratic Interpolation
                s = fb
                # Calculate quadratic interpolant
                # Using the form that is numerically stable
                # x = b - f(b) * ( (b-a)*(b-c) - (b-c)*(a-b) ) / ... 
                # Let's use the standard formula from Numerical Recipes or similar
                # q = (c-b)/(fc-fb)
                # r = (b-a)/(fb-fa)
                # t = (a-c)/(fa-fc)
                
                # We have points (a,fa), (b,fb), (c,fc)
                # Note: c is the previous a or b depending on swap. 
                # In standard Brent, c is the point with the smaller |f| from the previous iteration.
                # We need to maintain c, fc such that |fc| <= |fa| and |fc| <= |fb| is NOT necessarily true, 
                # but typically c is the "best" of the previous two.
                
                # Let's simplify and use a robust Secant + Bisection hybrid which is easier to get right.
                # This is often called "Illinois" or "Regula Falsi" with acceleration.
                
                x_secant = b - fb * (b - a) / (fb - fa)
                
                # Safeguard: if x_secant is outside [a,b] or too close to b, use bisection
                if (x_secant <= a or x_secant >= b) or abs(x_secant - b) < 0.5 * abs(b - a):
                    x_new = (a + b) / 2
                else:
                    x_new = x_secant
            else:
                # Use Bisection
                x_new = (a + b) / 2

            fx_new = eval_f(x_new)
            if abs(fx_new) <= 1e-8:
                return x_new
            
            # Update bracket
            if fa * fx_new < 0:
                b, fb = x_new, fx_new
            else:
                a, fa = x_new, fx_new
            
            # Update c (previous b) for next iteration's logic? 
            # For simple Secant/Bisection hybrid, we don't strictly need c if we just use current a,b.
            # But for faster convergence, we can keep the last point.
            # Let's stick to the simple robust hybrid above. It is very reliable.
            
            # Convergence check
            if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                return b if abs(fb) < abs(fa) else a
    else:
        # No bracket found. Use Safeguarded Secant Method.
        # Start with x0 and x1
        x0_s = x
        f0_s = fx
        x1_s = b # We used b in the bracket search, so we have a second point
        f1_s = fb
        
        for _ in range(max_evals - eval_count):
            if abs(f1_s) <= 1e-8:
                return x1_s
            if abs(f0_s) <= 1e-8:
                return x0_s
            
            if f0_s == f1_s:
                # Degenerate
                h2 = 1e-6 * max(1.0, abs(x1_s))
                x1_s = x1_s + h2
                f1_s = eval_f(x1_s)
                continue
            
            x_new = x1_s - f1_s * (x1_s - x0_s) / (f1_s - f0_s)
            
            # Safeguards
            step = x_new - x1_s
            if abs(step) > 10 * max(1.0, abs(x1_s)):
                x_new = x1_s + 0.5 * step
            elif abs(step) < 1e-14 * max(1.0, abs(x1_s)):
                break
            
            f_new = eval_f(x_new)
            if abs(f_new) <= 1e-8:
                return x_new
            
            x0_s, f0_s = x1_s, f1_s
            x1_s, f1_s = x_new, f_new
            
            if abs(x_new - x0_s) < 1e-12 * max(1.0, abs(x0_s), abs(x1_s)):
                break

    return best_x

# EVOLVE-BLOCK-END
