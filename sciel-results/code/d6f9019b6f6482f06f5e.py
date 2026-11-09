import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 for a smooth polynomial f.
    Uses bracketing followed by a Brent-like hybrid method (IQU + Bisection)
    for minimal function calls.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    max_calls = 480
    call_count = 0

    def time_ok():
        return (time.time() - start_time) < deadline and call_count < max_calls

    def f_call(x):
        nonlocal call_count
        call_count += 1
        return f(x)

    x = float(x0)
    if not math.isfinite(x):
        return x

    fx = f_call(x)
    if not math.isfinite(fx):
        return x

    if abs(fx) <= 1e-8:
        return x

    # Step 1: Find a bracket [a, b] with f(a)*f(b) < 0
    a = x
    fa = fx
    b = None
    fb = None

    # Try expanding right
    h = 1.0
    for _ in range(30):
        if not time_ok():
            break
        b = a + h
        fb = f_call(b)
        if not math.isfinite(fb):
            h *= 2.0
            continue
        if fa * fb < 0:
            break
        a = b
        fa = fb
        h *= 2.0

    if b is None or fa * fb >= 0:
        # Try expanding left from original x0
        a = x
        fa = fx
        b = None
        fb = None
        h = 1.0
        for _ in range(30):
            if not time_ok():
                break
            b = a - h
            fb = f_call(b)
            if not math.isfinite(fb):
                h *= 2.0
                continue
            if fa * fb < 0:
                break
            a = b
            fa = fb
            h *= 2.0

    if b is not None and fa * fb < 0:
        # We have a bracket [a, b] with f(a)*f(b) < 0
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Brent's method-like hybrid: Inverse Quadratic Interpolation + Bisection
        # Maintain a, b, c where f(a) and f(b) have opposite signs.
        # c is the previous b (or a) used for interpolation.
        # We want to converge to the root.
        
        c = a
        fc = fa
        mflag = True
        d = 0.0
        e = 0.0

        for _ in range(200):
            if not time_ok():
                break

            if abs(fb) <= 1e-8:
                return b
            if abs(fa) <= 1e-8:
                return a
            if abs(fc) <= 1e-8:
                return c

            # Check convergence
            if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                return 0.5 * (a + b)

            # Inverse quadratic interpolation if possible
            if mflag:
                d = (fb - fc) * (fb - fa)
                m = (b - c) * (fc - fa) - (b - a) * (fb - fc)
                mflag = False
            else:
                m = (b - c) * (fc - fa) - (b - a) * (fb - fc)

            # Check if interpolation is safe
            # Standard Brent's condition:
            # If 2*m < min(|d|, |b-a|*(fb-fa)) then use interpolation
            if 2 * m < min(abs(d), abs(b - a) * (fb - fa)):
                if abs(m) > 1e-300:
                    e_new = m / d
                    if not math.isfinite(e_new):
                        e_new = 0.5 * (b - a)
                else:
                    e_new = 0.5 * (b - a)
            else:
                e_new = 0.5 * (b - a)
                mflag = True

            # Check if step is too small; use bisection
            # e_new is the step from b.
            # We need e_new to be within (0.5*(a-b), 0.5*(b-a)) roughly, 
            # but standard check is if it's too close to 0 or too large.
            # Actually, the condition in Brent's is:
            # if e_new < 0.5*(a-b) and e_new > 0.5*(a-b)*-1 OR
            #    e_new < 0.5*(b-a) and e_new > 0.5*(b-a)*-1
            # This simplifies to: if e_new is not in the "safe" zone, bisect.
            
            half_ab = 0.5 * (b - a)
            if not (abs(e_new) < abs(half_ab) and e_new != 0):
                 # This logic is slightly tricky. Let's use the standard robust check:
                 # If the proposed step is not "reasonable", bisect.
                 pass
            
            # Simpler robust check for step size:
            # If the step is too small or too large relative to the bracket, bisect.
            # Standard Brent's:
            # if (e_new < 0.5 * (a - b) and e_new > 0.5 * (a - b) * -1) or \
            #    (e_new < 0.5 * (b - a) and e_new > 0.5 * (b - a) * -1):
            #     e_new = 0.5 * (b - a)
            #     mflag = True
            
            # Let's just ensure the next point is inside [a,b]
            x_next = b + e_new
            if x_next <= a or x_next >= b:
                x_next = 0.5 * (a + b)
                mflag = True

            # Update c
            c = a
            fc = fa

            f_next = f_call(x_next)
            if not math.isfinite(f_next):
                # Fallback to bisection
                x_next = 0.5 * (a + b)
                f_next = f_call(x_next)

            if abs(f_next) <= 1e-8:
                return x_next

            # Update bracket
            if fa * f_next < 0:
                b = x_next
                fb = f_next
            else:
                a = x_next
                fa = f_next

        # Return the better of the last points
        if abs(fa) < abs(fb):
            return a
        else:
            return b

    else:
        # No bracket found. Try Newton's method with finite differences.
        x = float(x0)
        fx = f_call(x)
        
        for _ in range(80):
            if not time_ok():
                break
            if abs(fx) <= 1e-8:
                return x
                
            h = 1e-7 * max(1.0, abs(x))
            fx_plus = f_call(x + h)
            d = (fx_plus - fx) / h
            
            if d == 0 or not math.isfinite(d):
                break
                
            x_new = x - fx / d
            
            if not math.isfinite(x_new):
                break
                
            # Damping
            if abs(x_new - x) > 10.0:
                x_new = x + 10.0 * math.copysign(1.0, x_new - x)
                
            x = x_new
            fx = f_call(x)
            
        return x
# EVOLVE-BLOCK-END
