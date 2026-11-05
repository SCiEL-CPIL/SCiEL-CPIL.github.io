"""
Strategy: Global black-box optimization of g(x) = |f(x)| using multi-start random restarts
and derivative-free safeguarded inverse quadratic interpolation (SIQI).
"""
import time
import math
import random

def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 145.0
    rng = random.Random(42)
    
    evals = 0
    best_x = float(x0)
    best_f = None
    best_abs = float('inf')
    
    def call_f(x):
        nonlocal evals, best_x, best_f, best_abs
        if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = f(x)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return None
        if not math.isfinite(val):
            return None
        abs_val = abs(val)
        if abs_val < best_abs:
            best_abs = abs_val
            best_x = x
            best_f = val
        return val
    
    def is_good(x, fx):
        if fx is None:
            return False
        return abs(fx) <= TOL
    
    def siqi_search(x0, x1, x2, max_iter=30):
        nonlocal evals
        # Ensure distinct points
        if x0 == x1 or x1 == x2 or x0 == x2:
            return x0, None
            
        fa = call_f(x0)
        fb = call_f(x1)
        fc = call_f(x2)
        
        if fa is None or fb is None or fc is None:
            # Fallback to best known
            candidates = [(x0, fa), (x1, fb), (x2, fc)]
            best_local = min(candidates, key=lambda t: abs(t[1]) if t[1] is not None else float('inf'))
            return best_local[0], best_local[1]
        
        if is_good(x0, fa): return x0, fa
        if is_good(x1, fb): return x1, fb
        if is_good(x2, fc): return x2, fc
        
        # Order points by |f|
        points = [(x0, fa), (x1, fb), (x2, fc)]
        points.sort(key=lambda t: abs(t[1]))
        
        # Invariant: a is best, b is second best, c is worst
        a, fa = points[0]
        b, fb = points[1]
        c, fc = points[2]
        
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 2:
                break
            if is_good(a, fa):
                return a, fa
                
            # Inverse quadratic interpolation
            # We want to find x such that f(x) = 0.
            # Using Lagrange interpolation on (a, fa), (b, fb), (c, fc)
            # The formula for the next point x_new:
            # x_new = (a^2 * fb * fc / ((a-b)*(a-c)) + b^2 * fa * fc / ((b-a)*(b-c)) + c^2 * fa * fb / ((c-a)*(c-b))) / 
            #         (fa * fc / ((a-b)*(a-c)) + fb * fc / ((b-a)*(b-c)) + fa * fb / ((c-a)*(c-b)))
            
            num = 0.0
            den = 0.0
            
            try:
                term1_num = a*a * fb * fc
                term1_den = (a-b) * (a-c)
                term2_num = b*b * fa * fc
                term2_den = (b-a) * (b-c)
                term3_num = c*c * fa * fb
                term3_den = (c-a) * (c-b)
                
                term1_den_num = fa * fc
                term2_den_num = fb * fc
                term3_den_num = fa * fb
                
                if term1_den != 0:
                    num += term1_num / term1_den
                    den += term1_den_num / term1_den
                if term2_den != 0:
                    num += term2_num / term2_den
                    den += term2_den_num / term2_den
                if term3_den != 0:
                    num += term3_num / term3_den
                    den += term3_den_num / term3_den
                    
                if abs(den) < 1e-30:
                    # Degenerate, use bisection-like step
                    x_new = (a + b) / 2.0
                else:
                    x_new = num / den
            except (ZeroDivisionError, OverflowError):
                x_new = (a + b) / 2.0
            
            if not math.isfinite(x_new):
                x_new = (a + b) / 2.0
                
            # Safeguard: x_new must be in the interval between the best and worst points
            # to ensure we are moving towards the root in a controlled manner
            min_bound = min(a, c)
            max_bound = max(a, c)
            
            if x_new <= min_bound or x_new >= max_bound:
                # Fall back to bisection between a and c (or a and b if c is too far)
                # Standard SIQI safeguard: if interpolation is outside, use bisection of the interval containing the root
                # Since we don't know sign, we bisect between the best and the worst to ensure progress
                x_new = (a + c) / 2.0
                
            fx_new = call_f(x_new)
            if fx_new is None:
                break
                
            if is_good(x_new, fx_new):
                return x_new, fx_new
                
            # Update points
            if abs(fx_new) < abs(fa):
                # New best
                c = b
                fc = fb
                b = a
                fb = fa
                a = x_new
                fa = fx_new
            elif abs(fx_new) < abs(fb):
                # New second best
                c = b
                fc = fb
                b = x_new
                fb = fx_new
            else:
                # New worst
                c = x_new
                fc = fx_new
                
            # Convergence check
            if abs(a - b) < 1e-12 * max(1.0, abs(a)):
                break
                
        return a, fa
    
    # Initial evaluation
    f0 = call_f(x0)
    if f0 is not None and is_good(x0, f0):
        return x0
        
    # Multi-start strategy
    # Generate a set of starting points
    # 1. x0
    # 2. x0 with small perturbations
    # 3. x0 with larger perturbations
    # 4. Random points in a wider range
    
    scale = max(1.0, abs(x0))
    
    # Phase 1: Local exploration around x0
    local_points = []
    for h in [1e-4, 1e-3, 1e-2, 1e-1]:
        local_points.append(x0 + h * scale)
        local_points.append(x0 - h * scale)
    
    # Phase 2: Global exploration
    global_points = []
    for _ in range(20):
        # Log-uniform distribution of step sizes
        log_h = rng.uniform(-6, 3) # h from 1e-6 to 1000
        h = 10**log_h * scale
        sign = 1 if rng.random() > 0.5 else -1
        global_points.append(x0 + sign * h)
        
    # Combine and sort by distance to x0 to prioritize local first
    all_points = [x0] + local_points + global_points
    all_points.sort(key=lambda x: abs(x - x0))
    
    # Evaluate initial points to find good triplets for SIQI
    # We need 3 distinct points to start SIQI
    # Let's evaluate a few to find a good initial bracket/triplet
    
    evaluated = []
    for x in all_points:
        if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 5:
            break
        fx = call_f(x)
        if fx is not None:
            evaluated.append((x, fx))
            if is_good(x, fx):
                return x
        if len(evaluated) >= 3:
            break
            
    if len(evaluated) < 3:
        # Not enough points, return best
        return best_x
        
    # Try SIQI from the best 3 points
    evaluated.sort(key=lambda t: abs(t[1]))
    best_triplet = evaluated[:3]
    
    x_res, f_res = siqi_search(best_triplet[0][0], best_triplet[1][0], best_triplet[2][0])
    if is_good(x_res, f_res):
        return x_res
        
    # If SIQI didn't converge, try more random restarts
    # Use remaining budget for random sampling + local SIQI
    
    while time.monotonic() < DEADLINE - 5 and evals < MAX_EVALS - 10:
        # Generate a new random point
        log_h = rng.uniform(-6, 3)
        h = 10**log_h * scale
        sign = 1 if rng.random() > 0.5 else -1
        x_new = x0 + sign * h
        
        # If we are close to a previous point, skip
        skip = False
        for x_old, _ in evaluated:
            if abs(x_new - x_old) < 1e-10 * max(1.0, abs(x_new)):
                skip = True
                break
        if skip:
            continue
            
        fx_new = call_f(x_new)
        if fx_new is None:
            continue
            
        if is_good(x_new, fx_new):
            return x_new
            
        evaluated.append((x_new, fx_new))
        
        # Keep only the top 10 best points to manage memory and focus search
        if len(evaluated) > 10:
            evaluated.sort(key=lambda t: abs(t[1]))
            evaluated = evaluated[:10]
            
        # Try SIQI with the best 3 points
        evaluated.sort(key=lambda t: abs(t[1]))
        if len(evaluated) >= 3:
            x_res, f_res = siqi_search(evaluated[0][0], evaluated[1][0], evaluated[2][0], max_iter=10)
            if is_good(x_res, f_res):
                return x_res
                
    # Final polish
    if best_f is not None and abs(best_f) < 1e-6:
        # Try a few more SIQI iterations from best_x
        h = 1e-5 * max(1.0, abs(best_x))
        x1 = best_x + h
        x2 = best_x - h
        x_res, f_res = siqi_search(best_x, x1, x2, max_iter=20)
        if is_good(x_res, f_res):
            return x_res
            
    return best_x
