"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    x0 = float(x0)
    
    # 1. Try to fit a cubic polynomial to 4 points around x0
    # Points: x0-2, x0-1, x0+1, x0+2
    # This uses 4 function calls.
    h = 1.0
    # If x0 is very large, relative step might be better, but polynomials 
    # are global. Fixed step is usually fine for "smooth polynomial" tasks 
    # unless x0 is huge. Let's use relative step if x0 is large.
    if abs(x0) > 10:
        h = abs(x0) * 0.1
    else:
        h = 1.0
    
    xs = [x0 - 2*h, x0 - h, x0 + h, x0 + 2*h]
    ys = []
    for x in xs:
        if time.time() >= deadline:
            return x0 # Fallback, shouldn't happen quickly
        y = f(x)
        if not math.isfinite(y):
            # If any point is non-finite, abort polynomial fit
            ys = None
            break
        ys.append(y)
    
    best_x = x0
    best_fx = f(x0) # 5th call
    best_abs = abs(best_fx)
    
    if best_abs <= 1e-8:
        return x0

    # Build history for interpolation: [(x, f(x)), ...]
    history = []
    if ys is not None and len(ys) == 4:
        for xi, yi in zip(xs, ys):
            history.append((xi, yi))
    history.append((x0, best_fx))
    
    if ys is not None and len(ys) == 4:
        # Fit cubic polynomial
        # P(x) = c0*x^3 + c1*x^2 + c2*x + c3
        # We have 4 points, so we can solve the linear system or use polyfit
        # np.polyfit returns coefficients in descending order: [a, b, c, d] for ax^3+bx^2+cx+d
        try:
            coeffs = np.polyfit(xs, ys, 3)
            # Find roots of the polynomial
            roots = np.roots(coeffs)
            
            # Filter real roots
            real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
            
            # Select the root closest to x0
            if real_roots:
                # Sort by distance to x0
                real_roots.sort(key=lambda r: abs(r - x0))
                
                # Check the closest root
                r_candidate = real_roots[0]
                
                # Verify candidate
                f_cand = f(r_candidate) # 6th call
                abs_cand = abs(f_cand)
                
                if math.isfinite(f_cand) and abs_cand < best_abs:
                    best_x = r_candidate
                    best_fx = f_cand
                    best_abs = abs_cand
                    history.append((r_candidate, f_cand))
                    
                    if abs_cand <= 1e-8:
                        return r_candidate

                # If the closest root wasn't good enough, maybe the second one is?
                # But usually the model is good. Let's try one more if we have budget.
                if len(real_roots) > 1:
                    r_candidate2 = real_roots[1]
                    f_cand2 = f(r_candidate2) # 7th call
                    abs_cand2 = abs(f_cand2)
                    
                    if math.isfinite(f_cand2) and abs_cand2 < best_abs:
                        best_x = r_candidate2
                        best_fx = f_cand2
                        best_abs = abs_cand2
                        history.append((r_candidate2, f_cand2))
                        
                        if abs_cand2 <= 1e-8:
                            return r_candidate2
        except Exception:
            pass # Ignore numerical issues in fitting

    # 2. Fallback: Hybrid Newton-Secant method starting from best_x
    # This is robust if the polynomial fit failed or was inaccurate.
    
    x = best_x
    fx = best_fx
    
    # Use Secant method as a robust alternative to Newton if derivative is bad
    # We have two points: (x, fx) and (x0, f(x0))
    x_prev = x0
    f_prev = f(x0) # We already computed this, but let's use it. 
                   # Wait, if best_x == x0, we have only 1 point. 
                   # If best_x != x0, we have 2 points.
    
    if best_x == x0:
        # Need another point for secant
        x_step = x0 + 0.001
        f_step = f(x_step)
        if math.isfinite(f_step):
            x_prev = x_step
            f_prev = f_step
            history.append((x_step, f_step))
        else:
            # Fallback to pure Newton with central diff
            x_prev = None
            f_prev = None
    else:
        x_prev = x0
        f_prev = f(x0)

    for i in range(30):
        if time.time() >= deadline:
            break
            
        if abs(fx) <= 1e-8:
            return x
            
        # Try quadratic interpolation if we have 3+ distinct points in history
        # This reuses existing points (0 extra calls) and achieves cubic convergence
        if len(history) >= 3 and x_prev is not None and f_prev is not None:
            # Get 3 most recent distinct points
            points = []
            seen = set()
            for px, py in reversed(history):
                key = (px, py)
                if key not in seen:
                    seen.add(key)
                    points.append((px, py))
                if len(points) == 3:
                    break
            
            if len(points) == 3:
                (x1, y1), (x2, y2), (x3, y3) = points
                # Quadratic interpolation: find x such that q(x) = 0
                # q(x) = a*x^2 + b*x + c where q(xi) = yi
                # Using Lagrange form or solving the system
                try:
                    # Solve for quadratic through 3 points
                    # [x1^2 x1 1] [a]   [y1]
                    # [x2^2 x2 1] [b] = [y2]
                    # [x3^2 x3 1] [c]   [y3]
                    A = np.array([[x1**2, x1, 1],
                                  [x2**2, x2, 1],
                                  [x3**2, x3, 1]])
                    b_vec = np.array([y1, y2, y3])
                    coeffs_q = np.linalg.solve(A, b_vec)
                    a_q, b_q, c_q = coeffs_q
                    
                    # Find roots of a_q*x^2 + b_q*x + c_q = 0
                    if abs(a_q) < 1e-15:
                        # Linear
                        if abs(b_q) > 1e-15:
                            x_interp = -c_q / b_q
                        else:
                            x_interp = None
                    else:
                        disc = b_q**2 - 4*a_q*c_q
                        if disc >= 0:
                            sqrt_disc = math.sqrt(disc)
                            r1 = (-b_q + sqrt_disc) / (2*a_q)
                            r2 = (-b_q - sqrt_disc) / (2*a_q)
                            # Pick root closest to current x
                            if abs(r1 - x) <= abs(r2 - x):
                                x_interp = r1
                            else:
                                x_interp = r2
                        else:
                            x_interp = None
                    
                    if x_interp is not None and math.isfinite(x_interp):
                        # Check if this is a new point
                        if x_interp != x and x_interp != x_prev:
                            f_interp = f(x_interp)
                            if math.isfinite(f_interp):
                                history.append((x_interp, f_interp))
                                if abs(f_interp) < abs(fx):
                                    # Update secant points
                                    x_prev = x
                                    f_prev = fx
                                    x = x_interp
                                    fx = f_interp
                                    if abs(fx) < best_abs:
                                        best_x = x
                                        best_abs = abs(fx)
                                    if abs(fx) <= 1e-8:
                                        return x
                                    continue
                except Exception:
                    pass
        
        if x_prev is not None and f_prev is not None and x != x_prev:
            # Secant step
            denom = fx - f_prev
            if denom != 0:
                x_new = x - fx * (x - x_prev) / denom
                if math.isfinite(x_new):
                    f_new = f(x_new)
                    if math.isfinite(f_new):
                        # Accept if it improves or if it's close
                        if abs(f_new) < abs(fx) or abs(f_new) < 1e-12:
                            # Shift points
                            x_prev = x
                            f_prev = fx
                            x = x_new
                            fx = f_new
                            history.append((x, fx))
                            if abs(fx) < best_abs:
                                best_x = x
                                best_abs = abs(fx)
                            continue
                        else:
                            # Secant step bad, try Newton
                            pass
            # If secant fails, fall through to Newton
        else:
            x_prev = None
            f_prev = None

        # Newton step with central difference
        h_newton = 1e-7 * max(1.0, abs(x))
        d = (f(x + h_newton) - f(x - h_newton)) / (2 * h_newton)
        
        if d == 0 or not math.isfinite(d):
            # Derivative is zero or bad. Try a small random perturbation or break
            break
            
        x_new = x - fx / d
        
        if not math.isfinite(x_new):
            break
            
        # Damping if step is too large
        if abs(x_new - x) > 10 * max(1.0, abs(x)):
            x_new = x + 0.5 * (x_new - x)
            
        f_new = f(x_new)
        
        if not math.isfinite(f_new):
            break
            
        if abs(f_new) < abs(fx):
            # Update secant points
            if x_prev is not None:
                x_prev = x
                f_prev = fx
            x = x_new
            fx = f_new
            history.append((x, fx))
            if abs(fx) < best_abs:
                best_x = x
                best_abs = abs(fx)
        else:
            # Newton diverging, break
            break

    # 3. Final high-precision polish: if we're close, use more Newton iterations
    # to reliably hit |f(x)| <= 1e-8
    if best_abs < 1e-4:  # We're reasonably close
        x = best_x
        fx = f(x)
        for _ in range(20):
            if time.time() >= deadline:
                break
            if abs(fx) <= 1e-8:
                return x
            h_newton = 1e-9 * max(1.0, abs(x))
            d = (f(x + h_newton) - f(x - h_newton)) / (2 * h_newton)
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            f_new = f(x_new)
            if not math.isfinite(f_new):
                break
            if abs(f_new) < abs(fx):
                x = x_new
                fx = f_new
            else:
                break

    return best_x
# EVOLVE-BLOCK-END
